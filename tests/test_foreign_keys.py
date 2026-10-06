"""Foreign key tests: definitions, commit-time enforcement, durability."""

import json
import os
import shutil
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from kvse import Engine, StorageError
from kvse.engine import ConstraintError
from kvse.http_app import create_server
from kvse.pager import HEADER_SIZE, PAGE_SIZE, crc32
from kvse.replica import Replica

ACCOUNTS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "owner", "type": "text", "nullable": False},
]
ORDERS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "account_id", "type": "int", "references": {"table": "accounts", "column": "id"}},
    {"name": "note", "type": "text"},
]


def make_engine(root):
    engine = Engine(root, now_ms=lambda: 1000)
    engine.create_table("accounts", ACCOUNTS, "id")
    engine.create_table("orders", ORDERS, "id")
    return engine


def read_state_image(pages_path, page_size=PAGE_SIZE):
    """Parse the meta record and state blob of a page file."""
    with open(pages_path, "rb") as fh:
        data = fh.read()

    def payload(page_id):
        block = data[page_id * page_size:(page_id + 1) * page_size]
        return block[HEADER_SIZE:]

    meta = json.loads(payload(0).rstrip(b"\x00").decode("utf-8"))
    blob = b"".join(payload(meta["state_page"] + i) for i in range(meta["state_pages"]))
    state = json.loads(blob[: meta["state_len"]].decode("utf-8"))
    return meta, state


def write_state_image(pages_path, meta, state, page_size=PAGE_SIZE):
    """Rewrite the state blob (and meta) of a page file with valid checksums."""
    blob = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_size = page_size - HEADER_SIZE
    count = max(1, -(-len(blob) // payload_size))
    meta = dict(meta, state_page=1 + meta["audit_pages"], state_pages=count, state_len=len(blob))
    with open(pages_path, "rb") as fh:
        data = bytearray(fh.read())

    def encode(page_id, content):
        block = content.ljust(payload_size, b"\x00")
        return struct.pack(">II", page_id, crc32(block)) + block

    pages = {0: json.dumps(meta, sort_keys=True).encode("utf-8")}
    for index in range(count):
        pages[meta["state_page"] + index] = blob[index * payload_size:(index + 1) * payload_size]
    for page_id, content in pages.items():
        data[page_id * page_size:(page_id + 1) * page_size] = encode(page_id, content)
    del data[(meta["state_page"] + count) * page_size:]
    with open(pages_path, "wb") as fh:
        fh.write(bytes(data))
    return meta


class ForeignKeyDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-def-")
        self.engine = make_engine(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_references_are_stored_and_reported_verbatim(self):
        info = self.engine.table_info("orders")
        column = [c for c in info["columns"] if c["name"] == "account_id"][0]
        self.assertEqual(column["references"], {"table": "accounts", "column": "id"})
        # mutating the reported object must not corrupt the catalog
        column["references"]["table"] = "nope"
        again = [c for c in self.engine.table_info("orders")["columns"] if c["name"] == "account_id"][0]
        self.assertEqual(again["references"], {"table": "accounts", "column": "id"})
        # columns without a reference carry no references key at all
        for info in (self.engine.table_info("accounts"), self.engine.table_info("orders")):
            for column in info["columns"]:
                if column["name"] != "account_id":
                    self.assertNotIn("references", column)

    def test_create_result_carries_references(self):
        info = self.engine.create_table("payments", [
            {"name": "id", "type": "int"},
            {"name": "order_id", "type": "int", "references": {"table": "orders", "column": "id"}},
        ], "id")
        column = [c for c in info["columns"] if c["name"] == "order_id"][0]
        self.assertEqual(column["references"], {"table": "orders", "column": "id"})

    def test_invalid_definitions_are_rejected(self):
        base = [{"name": "id", "type": "int"}]
        bad_refs = [
            "accounts",                                                  # not an object
            {"table": "accounts"},                                       # missing column
            {"column": "id"},                                            # missing table
            {"table": "accounts", "column": "id", "on_delete": "cascade"},  # extra key
            {"table": "", "column": "id"},                               # blank table
            {"table": "accounts", "column": 5},                          # non-string column
            {"table": "missing", "column": "id"},                        # unknown table
            {"table": "accounts", "column": "owner"},                    # not the primary key
            {"table": "accounts", "column": "nope"},                     # unknown target column
            {"table": "child", "column": "id"},                          # self reference: not yet existing
        ]
        for ref in bad_refs:
            columns = base + [{"name": "aid", "type": "int", "references": ref}]
            with self.assertRaises(StorageError, msg=repr(ref)):
                self.engine.create_table("child", columns, "id")
        columns = base + [{"name": "aid", "type": "text", "references": {"table": "accounts", "column": "id"}}]
        with self.assertRaises(StorageError):  # type mismatch
            self.engine.create_table("child", columns, "id")
        self.assertEqual(self.engine.list_tables(), ["accounts", "orders"])


class ForeignKeyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-run-")
        self.engine = make_engine(self.root)
        self.engine.insert("accounts", {"id": 1, "owner": "ann"})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_null_never_triggers_a_lookup(self):
        row = self.engine.insert("orders", {"id": 1, "account_id": None})
        self.assertIsNone(row["account_id"])
        row = self.engine.insert("orders", {"id": 2})
        self.assertIsNone(row["account_id"])

    def test_insert_and_update_checks(self):
        self.engine.insert("orders", {"id": 1, "account_id": 1})
        with self.assertRaises(ConstraintError):
            self.engine.insert("orders", {"id": 2, "account_id": 99})
        self.assertIsNone(self.engine.get("orders", 2))
        with self.assertRaises(ConstraintError):
            self.engine.update("orders", 1, {"account_id": 99})
        self.assertEqual(self.engine.get("orders", 1)["account_id"], 1)
        self.engine.update("orders", 1, {"account_id": None})
        self.assertIsNone(self.engine.get("orders", 1)["account_id"])

    def test_a_failed_commit_leaves_no_partial_result(self):
        lsn, audit = self.engine.lsn, self.engine.audit()
        with self.assertRaises(ConstraintError):
            self.engine.insert("orders", {"id": 5, "account_id": 42})
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.audit(), audit)
        self.assertEqual(self.engine.scan("orders"), [])
        self.assertTrue(self.engine.verify()["ok"])

    def test_delete_of_a_referenced_parent_is_rejected(self):
        self.engine.insert("orders", {"id": 1, "account_id": 1})
        with self.assertRaises(ConstraintError):
            self.engine.delete("accounts", 1)
        self.assertEqual(self.engine.get("accounts", 1)["owner"], "ann")
        self.engine.delete("orders", 1)
        self.engine.delete("accounts", 1)  # unreferenced now
        self.assertIsNone(self.engine.get("accounts", 1))

    def test_parent_and_child_in_one_transaction_any_order(self):
        tx = self.engine.begin()
        tx.insert("orders", {"id": 1, "account_id": 7})  # child first
        tx.insert("accounts", {"id": 7, "owner": "bob"})
        tx.commit()
        self.assertEqual(self.engine.get("orders", 1)["account_id"], 7)

        tx = self.engine.begin()
        tx.delete("orders", 1)  # child first
        tx.delete("accounts", 7)
        tx.commit()
        self.assertIsNone(self.engine.get("accounts", 7))

    def test_a_dangling_batch_rolls_back_completely(self):
        lsn = self.engine.lsn
        with self.assertRaises(ConstraintError):
            self.engine.transaction([
                {"op": "insert", "table": "accounts", "row": {"id": 2, "owner": "bob"}},
                {"op": "insert", "table": "orders", "row": {"id": 1, "account_id": 2}},
                {"op": "delete", "table": "accounts", "pk": 2},
            ])
        self.assertIsNone(self.engine.get("accounts", 2))
        self.assertEqual(self.engine.scan("orders"), [])
        self.assertEqual(self.engine.lsn, lsn)
        self.assertTrue(self.engine.verify()["ok"])

    def test_failed_commit_terminates_the_transaction(self):
        tx = self.engine.begin()
        tx.insert("orders", {"id": 1, "account_id": 42})
        with self.assertRaises(ConstraintError):
            tx.commit()
        self.assertEqual(tx.state, "rolled_back")
        with self.assertRaises(StorageError):
            tx.insert("orders", {"id": 2, "account_id": 1})
        self.assertEqual(self.engine.scan("orders"), [])

    def test_savepoint_rollback_restores_referential_integrity(self):
        tx = self.engine.begin()
        tx.insert("orders", {"id": 1, "account_id": 1})
        tx.savepoint("s")
        tx.insert("orders", {"id": 2, "account_id": 42})
        tx.rollback_to("s")
        tx.insert("orders", {"id": 2, "account_id": 1})
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.scan("orders")], [1, 2])

    def test_snapshot_visibility_governs_the_parent_lookup(self):
        # a parent committed after the reader's snapshot is not visible to it
        tx = self.engine.begin()
        self.engine.insert("accounts", {"id": 3, "owner": "cid"})
        tx.insert("orders", {"id": 1, "account_id": 3})
        with self.assertRaises(ConstraintError):
            tx.commit()
        self.assertEqual(self.engine.scan("orders"), [])

    def test_serializable_transactions_check_foreign_keys_too(self):
        tx = self.engine.begin(isolation="serializable")
        tx.insert("orders", {"id": 1, "account_id": 1})
        tx.commit()
        self.assertEqual(self.engine.get("orders", 1)["account_id"], 1)
        tx = self.engine.begin(isolation="serializable")
        tx.insert("orders", {"id": 2, "account_id": 42})
        with self.assertRaises(ConstraintError):
            tx.commit()

    def test_reads_are_unaffected_by_foreign_keys(self):
        self.engine.create_index("orders", "account_id")
        self.engine.insert("orders", {"id": 1, "account_id": 1, "note": "a"})
        self.engine.insert("orders", {"id": 2, "note": "b"})
        self.assertEqual([r["id"] for r in self.engine.scan("orders")], [1, 2])
        self.assertEqual(self.engine.index_get("orders", "account_id", 1)[0]["id"], 1)
        self.assertEqual(
            self.engine.explain("orders", where=[("account_id", "=", 1)]),
            {"access": "index", "index": "orders_account_id_idx"},
        )
        result = self.engine.query("orders", where=[("account_id", "=", 1)])
        self.assertEqual([r["id"] for r in result["rows"]], [1])
        view = self.engine.readonly_view()
        self.assertEqual(len(view.scan("orders")), 2)
        with self.assertRaises(StorageError):
            view.insert("orders", {"id": 3, "account_id": 1})


class ForeignKeyDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-dur-")
        self.engine = make_engine(self.root)
        self.engine.insert("accounts", {"id": 1, "owner": "ann"})
        self.engine.insert("orders", {"id": 1, "account_id": 1})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def references(self, engine=None):
        info = (engine or self.engine).table_info("orders")
        return [c for c in info["columns"] if c["name"] == "account_id"][0].get("references")

    def test_reopen_keeps_definitions_and_enforcement(self):
        self.engine.reopen()
        self.assertEqual(self.references(), {"table": "accounts", "column": "id"})
        with self.assertRaises(ConstraintError):
            self.engine.insert("orders", {"id": 2, "account_id": 42})
        with self.assertRaises(ConstraintError):
            self.engine.delete("accounts", 1)

    def test_crash_recovery_keeps_definitions(self):
        with open(self.engine.pager.data_path, "wb"):
            pass  # every page lost; only the WAL survives
        self.engine.reopen()
        self.assertEqual(self.references(), {"table": "accounts", "column": "id"})
        self.assertEqual(self.engine.get("orders", 1)["account_id"], 1)

    def test_backup_and_point_in_time_restore(self):
        self.engine.insert("accounts", {"id": 2, "owner": "bob"})
        lsn = self.engine.lsn
        self.engine.insert("orders", {"id": 2, "account_id": 2})
        backup = os.path.join(self.root, "backup")
        self.engine.backup(backup)

        self.engine.restore(backup, to_lsn=lsn)
        self.assertEqual(self.references(), {"table": "accounts", "column": "id"})
        self.assertIsNone(self.engine.get("orders", 2))
        with self.assertRaises(ConstraintError):
            self.engine.insert("orders", {"id": 3, "account_id": 42})

        self.engine.restore(backup)
        self.assertEqual(self.references(), {"table": "accounts", "column": "id"})
        self.assertEqual(self.engine.get("orders", 2)["account_id"], 2)

    def corrupt_backup_into_dangling(self, backup):
        """Remove the referenced account from the backup's state image."""
        pages = os.path.join(backup, "data.pages")
        meta, state = read_state_image(pages)
        state["tables"]["accounts"]["rows"] = []
        write_state_image(pages, meta, state)
        with open(os.path.join(backup, "wal.log"), "wb"):
            pass  # no WAL records: the crafted image is all there is

    def test_restore_rejects_dangling_data_without_replacing_state(self):
        backup = os.path.join(self.root, "backup")
        self.engine.backup(backup)
        self.corrupt_backup_into_dangling(backup)
        lsn = self.engine.lsn
        with self.assertRaises(StorageError):
            self.engine.restore(backup)
        # the previous state is untouched
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.get("accounts", 1)["owner"], "ann")
        self.assertEqual(self.engine.get("orders", 1)["account_id"], 1)
        self.assertTrue(self.engine.verify()["ok"])

    def test_reopen_rejects_dangling_data(self):
        other = os.path.join(self.root, "other")
        engine = Engine(other, now_ms=lambda: 1000)
        engine.create_table("accounts", ACCOUNTS, "id")
        engine.create_table("orders", ORDERS, "id")
        engine.insert("accounts", {"id": 1, "owner": "ann"})
        engine.insert("orders", {"id": 1, "account_id": 1})
        pages = engine.pager.data_path
        meta, state = read_state_image(pages)
        state["tables"]["accounts"]["rows"] = []
        write_state_image(pages, meta, state)
        with open(engine.pager.wal_path, "wb"):
            pass
        with self.assertRaises(StorageError):
            engine.reopen()

    def test_replica_sync_preserves_references_and_refuses_writes(self):
        replica_dir = os.path.join(self.root, "replica")
        replica = Replica.create(self.engine, replica_dir)
        info = replica.table_info("orders")
        column = [c for c in info["columns"] if c["name"] == "account_id"][0]
        self.assertEqual(column["references"], {"table": "accounts", "column": "id"})
        self.engine.insert("accounts", {"id": 2, "owner": "bob"})
        self.engine.insert("orders", {"id": 2, "account_id": 2})
        replica.sync()
        self.assertEqual(replica.get("orders", 2)["account_id"], 2)
        session = replica.read_session()
        try:
            column = [c for c in session.table_info("orders")["columns"] if c["name"] == "account_id"][0]
            self.assertEqual(column["references"], {"table": "accounts", "column": "id"})
            with self.assertRaises(StorageError):
                session.insert("orders", {"id": 3})
        finally:
            session.close()
        with self.assertRaises(StorageError):
            replica.insert("orders", {"id": 3, "account_id": 1})
        with self.assertRaises(StorageError):
            replica.delete("accounts", 1)
        replica.reopen()
        column = [c for c in replica.table_info("orders")["columns"] if c["name"] == "account_id"][0]
        self.assertEqual(column["references"], {"table": "accounts", "column": "id"})

    def test_replica_sync_rejects_a_dangling_source_and_keeps_old_state(self):
        replica_dir = os.path.join(self.root, "replica")
        replica = Replica.create(self.engine, replica_dir)
        applied = replica.applied_lsn
        # corrupt the primary's image into a dangling state at a newer lsn
        pages = self.engine.pager.data_path
        meta, state = read_state_image(pages)
        state["tables"]["accounts"]["rows"] = []
        write_state_image(pages, dict(meta, lsn=meta["lsn"] + 1), state)
        with open(self.engine.pager.wal_path, "wb"):
            pass
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("orders", 1)["account_id"], 1)
        self.assertEqual(replica.get("accounts", 1)["owner"], "ann")


class ForeignKeyHttpTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-http-")
        self.engine = Engine(self.root, now_ms=lambda: 7000)
        self.server = create_server(self.engine, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.root, ignore_errors=True)

    def call(self, method, path, body=None):
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=None if body is None else json.dumps(body).encode("utf-8"),
            method=method,
        )
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def create_tables(self):
        status, _ = self.call("POST", "/v1/tables", {
            "name": "accounts", "columns": ACCOUNTS, "primary_key": "id",
        })
        self.assertEqual(status, 201)
        status, body = self.call("POST", "/v1/tables", {
            "name": "orders", "columns": ORDERS, "primary_key": "id",
        })
        self.assertEqual(status, 201)
        return body

    def test_create_table_with_references(self):
        body = self.create_tables()
        column = [c for c in body["columns"] if c["name"] == "account_id"][0]
        self.assertEqual(column["references"], {"table": "accounts", "column": "id"})

    def test_create_table_with_a_bad_reference_is_a_400(self):
        self.call("POST", "/v1/tables", {
            "name": "accounts", "columns": ACCOUNTS, "primary_key": "id",
        })
        status, body = self.call("POST", "/v1/tables", {
            "name": "orders",
            "columns": [{"name": "id", "type": "int"},
                        {"name": "aid", "type": "int",
                         "references": {"table": "accounts", "column": "owner"}}],
            "primary_key": "id",
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.assertEqual(self.call("GET", "/v1/tables")[1]["tables"], ["accounts"])

    def test_dangling_writes_are_409_and_atomic(self):
        self.create_tables()
        self.call("POST", "/v1/tables/accounts/rows", {"rows": [{"id": 1, "owner": "ann"}]})

        status, body = self.call("POST", "/v1/tables/orders/rows", {
            "rows": [{"id": 1, "account_id": 1}, {"id": 2, "account_id": 42}],
        })
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        self.assertEqual(self.call("GET", "/v1/tables/orders/rows/1")[0], 404)

        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "orders", "row": {"id": 3, "account_id": 1}},
            {"op": "delete", "table": "accounts", "pk": 1},
        ]})
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        self.assertEqual(self.call("GET", "/v1/tables/accounts/rows/1")[0], 200)
        self.assertEqual(self.call("GET", "/v1/tables/orders/rows/3")[0], 404)

        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "orders", "row": {"id": 3, "account_id": 1}},
        ]})
        self.assertEqual(status, 200)
        self.assertTrue(body["committed"])


if __name__ == "__main__":
    unittest.main()
