"""Foreign key tests: definitions, runtime checks, atomicity, recovery."""

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
from kvse.pager import HEADER_SIZE, crc32
from kvse.replica import Replica

ACCOUNTS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "owner", "type": "text", "nullable": False},
]

ORDERS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "account_id", "type": "int", "references": {"table": "accounts", "column": "id"}},
    {"name": "label", "type": "text"},
]


def rewrite_state(root, mutate, page_size=4096, meta_mutate=None):
    """Rewrite the persisted state blob of the engine directory ``root``.

    Applies ``mutate`` to the decoded state, writes the pages back with valid
    headers and truncates the WAL (a checkpoint), so the next open loads
    exactly the mutated image.  ``meta_mutate`` may adjust the meta record.
    """
    data_path = os.path.join(root, "data.pages")
    with open(data_path, "rb") as fh:
        data = fh.read()
    payload_size = page_size - HEADER_SIZE

    def payload(page_id):
        start = page_id * page_size + HEADER_SIZE
        return data[start:start + payload_size]

    meta = json.loads(payload(0).rstrip(b"\x00").decode("utf-8"))
    blob = b"".join(payload(meta["state_page"] + i) for i in range(meta["state_pages"]))
    state = json.loads(blob[: meta["state_len"]].decode("utf-8"))
    mutate(state)
    if meta_mutate is not None:
        meta_mutate(meta)
    new_blob = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    count = max(1, -(-len(new_blob)) // payload_size)
    meta["state_pages"] = count
    meta["state_len"] = len(new_blob)
    pages = {0: json.dumps(meta, sort_keys=True).encode("utf-8")}
    for index in range(count):
        pages[meta["state_page"] + index] = new_blob[index * payload_size:(index + 1) * payload_size]
    with open(data_path, "r+b") as fh:
        for page_id, page_payload in pages.items():
            block = page_payload.ljust(payload_size, b"\x00")
            fh.seek(page_id * page_size)
            fh.write(struct.pack(">II", page_id, crc32(block)) + block)
        fh.truncate((max(pages) + 1) * page_size)
        fh.flush()
        os.fsync(fh.fileno())
    with open(os.path.join(root, "wal.log"), "wb") as fh:
        fh.flush()
        os.fsync(fh.fileno())


class ForeignKeyDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-def-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("accounts", ACCOUNTS, "id")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def references(self, target):
        columns = [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "account_id", "type": "int", "references": target},
        ]
        return columns

    def test_valid_reference_is_stored_and_returned(self):
        info = self.engine.create_table("orders", ORDERS, "id")
        ref = {"table": "accounts", "column": "id"}
        by_name = {col["name"]: col for col in info["columns"]}
        self.assertEqual(by_name["account_id"]["references"], ref)
        self.assertNotIn("references", by_name["id"])
        info = self.engine.table_info("orders")
        by_name = {col["name"]: col for col in info["columns"]}
        self.assertEqual(by_name["account_id"]["references"], ref)
        # Mutating the returned object must not touch the catalog.
        by_name["account_id"]["references"]["table"] = "nope"
        info = self.engine.table_info("orders")
        self.assertEqual(
            {c["name"]: c for c in info["columns"]}["account_id"]["references"], ref
        )

    def test_columns_without_reference_stay_plain(self):
        info = self.engine.table_info("accounts")
        for column in info["columns"]:
            self.assertNotIn("references", column)

    def test_invalid_reference_shapes(self):
        bad = [
            "accounts.id",
            None,
            {"table": "accounts"},
            {"column": "id"},
            {"table": "accounts", "column": "id", "on_delete": "cascade"},
            {"table": "", "column": "id"},
            {"table": "accounts", "column": ""},
            {"table": 1, "column": "id"},
            {"table": "accounts", "column": None},
        ]
        for target in bad:
            with self.assertRaises(StorageError, msg=repr(target)):
                self.engine.create_table("orders", self.references(target), "id")
        self.assertEqual(self.engine.list_tables(), ["accounts"])

    def test_invalid_reference_targets(self):
        bad = [
            {"table": "nope", "column": "id"},       # unknown table
            {"table": "accounts", "column": "owner"},  # not the primary key
            {"table": "accounts", "column": "nope"},  # unknown column
            {"table": "orders", "column": "id"},      # self-reference: not yet existing
        ]
        for target in bad:
            with self.assertRaises(StorageError, msg=repr(target)):
                self.engine.create_table("orders", self.references(target), "id")
        self.assertEqual(self.engine.list_tables(), ["accounts"])

    def test_reference_type_must_match(self):
        columns = [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "account_id", "type": "text",
             "references": {"table": "accounts", "column": "id"}},
        ]
        with self.assertRaises(StorageError):
            self.engine.create_table("orders", columns, "id")
        self.assertEqual(self.engine.list_tables(), ["accounts"])

    def test_text_and_bool_primary_keys_can_be_referenced(self):
        self.engine.create_table("tags", [
            {"name": "name", "type": "text", "nullable": False},
        ], "name")
        info = self.engine.create_table("tag_links", [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "tag", "type": "text",
             "references": {"table": "tags", "column": "name"}},
        ], "id")
        by_name = {col["name"]: col for col in info["columns"]}
        self.assertEqual(by_name["tag"]["references"], {"table": "tags", "column": "name"})


class ForeignKeyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-run-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("accounts", ACCOUNTS, "id")
        self.engine.create_table("orders", ORDERS, "id")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def account(self, pk, owner="ann"):
        return self.engine.insert("accounts", {"id": pk, "owner": owner})

    def order(self, pk, account_id, label="o"):
        row = {"id": pk, "label": label}
        if account_id is not None:
            row["account_id"] = account_id
        return self.engine.insert("orders", row)

    def test_insert_with_missing_parent_is_rejected(self):
        with self.assertRaises(ConstraintError):
            self.order(1, 99)
        self.assertEqual(self.engine.scan("orders"), [])

    def test_null_foreign_key_never_looks_up(self):
        self.order(1, None)
        self.assertIsNone(self.engine.get("orders", 1)["account_id"])

    def test_insert_and_update_with_existing_parent(self):
        self.account(1)
        self.order(10, 1)
        self.account(2)
        updated = self.engine.update("orders", 10, {"account_id": 2})
        self.assertEqual(updated["account_id"], 2)
        with self.assertRaises(ConstraintError):
            self.engine.update("orders", 10, {"account_id": 77})
        self.assertEqual(self.engine.get("orders", 10)["account_id"], 2)
        cleared = self.engine.update("orders", 10, {"account_id": None})
        self.assertIsNone(cleared["account_id"])

    def test_parent_and_child_in_one_transaction(self):
        tx = self.engine.begin()
        tx.insert("accounts", {"id": 1, "owner": "ann"})
        tx.insert("orders", {"id": 10, "account_id": 1})  # sees the earlier write
        tx.commit()
        self.assertEqual(self.engine.get("orders", 10)["account_id"], 1)

    def test_child_before_parent_in_one_transaction_is_rejected(self):
        tx = self.engine.begin()
        with self.assertRaises(ConstraintError):
            tx.insert("orders", {"id": 10, "account_id": 1})
        tx.insert("accounts", {"id": 1, "owner": "ann"})
        tx.insert("orders", {"id": 10, "account_id": 1})  # parent now visible
        tx.commit()
        self.assertEqual(len(self.engine.scan("orders")), 1)

    def test_delete_of_referenced_parent_is_rejected(self):
        self.account(1)
        self.order(10, 1)
        with self.assertRaises(ConstraintError):
            self.engine.delete("accounts", 1)
        self.assertIsNotNone(self.engine.get("accounts", 1))
        self.engine.delete("orders", 10)
        self.engine.delete("accounts", 1)
        self.assertIsNone(self.engine.get("accounts", 1))

    def test_child_then_parent_delete_in_one_transaction(self):
        self.account(1)
        self.order(10, 1)
        tx = self.engine.begin()
        tx.delete("orders", 10)
        tx.delete("accounts", 1)
        tx.commit()
        self.assertEqual(self.engine.scan("accounts"), [])

    def test_parent_delete_before_child_delete_is_rejected(self):
        self.account(1)
        self.order(10, 1)
        tx = self.engine.begin()
        with self.assertRaises(ConstraintError):
            tx.delete("accounts", 1)
        tx.rollback()
        self.assertIsNotNone(self.engine.get("accounts", 1))

    def test_failed_transaction_leaves_no_trace(self):
        self.account(1)
        lsn = self.engine.lsn
        audit = len(self.engine.audit())
        tx = self.engine.begin()
        tx.insert("accounts", {"id": 2, "owner": "bob"})
        tx.insert("orders", {"id": 10, "account_id": 2})
        with self.assertRaises(ConstraintError):
            tx.delete("accounts", 2)  # still referenced by the transaction's own order
        tx.rollback()
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(len(self.engine.audit()), audit)
        self.assertEqual(self.engine.scan("orders"), [])
        self.assertEqual(len(self.engine.scan("accounts")), 1)

    def test_batch_transaction_is_atomic(self):
        self.account(1)
        lsn = self.engine.lsn
        audit = len(self.engine.audit())
        ops = [
            {"op": "insert", "table": "accounts", "row": {"id": 2, "owner": "bob"}},
            {"op": "insert", "table": "orders", "row": {"id": 10, "account_id": 2}},
            {"op": "insert", "table": "orders", "row": {"id": 11, "account_id": 99}},
        ]
        with self.assertRaises(ConstraintError):
            self.engine.transaction(ops)
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(len(self.engine.audit()), audit)
        self.assertEqual(self.engine.scan("orders"), [])
        self.assertIsNone(self.engine.get("accounts", 2))

    def test_savepoint_rollback_drops_deferred_checks(self):
        tx = self.engine.begin()
        tx.insert("accounts", {"id": 1, "owner": "ann"})
        tx.savepoint("sp")
        tx.insert("orders", {"id": 10, "account_id": 1})
        tx.rollback_to("sp")  # the child row and its deferred check are gone
        tx.delete("accounts", 1)  # nothing references it any more
        tx.commit()
        self.assertEqual(self.engine.scan("accounts"), [])
        self.assertEqual(self.engine.scan("orders"), [])

    def test_commit_time_check_catches_concurrent_parent_delete(self):
        self.account(1)
        tx_child = self.engine.begin()
        tx_child.insert("orders", {"id": 10, "account_id": 1})
        tx_parent = self.engine.begin()
        tx_parent.delete("accounts", 1)  # the uncommitted child is invisible
        tx_parent.commit()
        lsn = self.engine.lsn
        audit = len(self.engine.audit())
        with self.assertRaises(ConstraintError):
            tx_child.commit()
        # The whole transaction is gone: no row, no audit record, no lsn move.
        self.assertEqual(tx_child.state, "rolled_back")
        self.assertEqual(self.engine.scan("orders"), [])
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(len(self.engine.audit()), audit)
        self.assertIsNone(self.engine.get("accounts", 1))

    def test_commit_time_check_catches_concurrent_child_insert(self):
        self.account(1)
        tx_parent = self.engine.begin()
        tx_parent.delete("accounts", 1)
        tx_child = self.engine.begin()
        tx_child.insert("orders", {"id": 10, "account_id": 1})  # parent visible in its snapshot
        tx_child.commit()
        lsn = self.engine.lsn
        with self.assertRaises(ConstraintError):
            tx_parent.commit()
        self.assertEqual(tx_parent.state, "rolled_back")
        self.assertIsNotNone(self.engine.get("accounts", 1))
        self.assertEqual(self.engine.lsn, lsn)

    def test_reads_are_not_changed_by_foreign_keys(self):
        self.account(1)
        self.account(2, "bob")
        self.order(10, 1)
        self.order(11, 2)
        self.assertEqual([r["id"] for r in self.engine.scan("orders")], [10, 11])
        result = self.engine.query("orders", where=[["account_id", "=", 1]])
        self.assertEqual([r["id"] for r in result["rows"]], [10])
        self.assertEqual(self.engine.explain("orders"), {"access": "scan", "index": None})
        view = self.engine.readonly_view()
        self.assertEqual(view.get("orders", 10)["account_id"], 1)

    def test_serializable_transaction_with_fk_writes(self):
        self.account(1)
        tx = self.engine.begin(isolation="serializable")
        tx.scan("accounts")
        tx.insert("orders", {"id": 10, "account_id": 1})
        tx.commit()
        self.assertEqual(self.engine.get("orders", 10)["account_id"], 1)


class ForeignKeyPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-fk-persist-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("accounts", ACCOUNTS, "id")
        self.engine.create_table("orders", ORDERS, "id")
        self.engine.insert("accounts", {"id": 1, "owner": "ann"})
        self.engine.insert("orders", {"id": 10, "account_id": 1})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def references_of(self, engine, table="orders", column="account_id"):
        info = engine.table_info(table)
        return {c["name"]: c for c in info["columns"]}[column].get("references")

    def test_reopen_keeps_definitions_and_enforcement(self):
        self.engine.reopen()
        self.assertEqual(
            self.references_of(self.engine), {"table": "accounts", "column": "id"}
        )
        with self.assertRaises(ConstraintError):
            self.engine.insert("orders", {"id": 11, "account_id": 99})
        with self.assertRaises(ConstraintError):
            self.engine.delete("accounts", 1)

    def test_backup_and_restore_keep_references(self):
        backup = tempfile.mkdtemp(prefix="kvse-fk-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)
        other = tempfile.mkdtemp(prefix="kvse-fk-other-")
        self.addCleanup(shutil.rmtree, other, True)
        restored = Engine(other, now_ms=lambda: 1000)
        restored.restore(backup)
        self.assertEqual(
            self.references_of(restored), {"table": "accounts", "column": "id"}
        )
        self.assertEqual(restored.get("orders", 10)["account_id"], 1)
        with self.assertRaises(ConstraintError):
            restored.delete("accounts", 1)

    def test_restore_to_lsn_keeps_consistent_state(self):
        first_lsn = self.engine.lsn
        self.engine.insert("accounts", {"id": 2, "owner": "bob"})
        self.engine.insert("orders", {"id": 11, "account_id": 2})
        backup = tempfile.mkdtemp(prefix="kvse-fk-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)
        other = tempfile.mkdtemp(prefix="kvse-fk-other-")
        self.addCleanup(shutil.rmtree, other, True)
        restored = Engine(other, now_ms=lambda: 1000)
        restored.restore(backup, to_lsn=first_lsn)
        self.assertEqual(restored.scan("accounts"), [{"id": 1, "owner": "ann"}])
        self.assertEqual(len(restored.scan("orders")), 1)
        self.assertEqual(
            self.references_of(restored), {"table": "accounts", "column": "id"}
        )

    def test_replica_sync_keeps_references_and_refuses_writes(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-fk-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        self.assertEqual(
            self.references_of(replica), {"table": "accounts", "column": "id"}
        )
        self.engine.insert("accounts", {"id": 2, "owner": "bob"})
        self.engine.insert("orders", {"id": 11, "account_id": 2})
        replica.sync()
        self.assertEqual(replica.get("orders", 11)["account_id"], 2)
        session = replica.read_session()
        self.assertEqual(
            self.references_of(session), {"table": "accounts", "column": "id"}
        )
        session.close()
        with self.assertRaises(StorageError):
            replica.insert("orders", {"id": 12, "account_id": 2})

    def test_dangling_loaded_state_is_rejected(self):
        def dangle(state):
            state["tables"]["orders"]["rows"].append([99, {"id": 99, "account_id": 77, "label": "x"}])

        rewrite_state(self.root, dangle)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_missing_reference_target_is_rejected(self):
        def missing(state):
            for column in state["tables"]["orders"]["columns"]:
                if column["name"] == "account_id":
                    column["references"] = {"table": "nope", "column": "id"}

        rewrite_state(self.root, missing)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_type_mismatched_reference_is_rejected(self):
        def mismatch(state):
            for column in state["tables"]["orders"]["columns"]:
                if column["name"] == "account_id":
                    column["type"] = "text"

        rewrite_state(self.root, mismatch)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_sync_of_dangling_source_keeps_replica_state(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-fk-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        applied = replica.applied_lsn

        def dangle(state):
            state["tables"]["orders"]["rows"].append([99, {"id": 99, "account_id": 77, "label": "x"}])

        rewrite_state(
            self.root, dangle,  # corrupt the primary's checkpointed image
            meta_mutate=lambda meta: meta.update(lsn=meta["lsn"] + 10),
        )
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("orders", 10)["account_id"], 1)
        self.assertEqual(
            self.references_of(replica), {"table": "accounts", "column": "id"}
        )

    def test_restore_of_dangling_backup_keeps_current_state(self):
        backup = tempfile.mkdtemp(prefix="kvse-fk-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)

        def dangle(state):
            state["tables"]["orders"]["rows"].append([99, {"id": 99, "account_id": 77, "label": "x"}])

        rewrite_state(backup, dangle)
        lsn = self.engine.lsn
        with self.assertRaises(StorageError):
            self.engine.restore(backup)
        # The live engine is untouched.
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.get("orders", 10)["account_id"], 1)
        self.assertEqual(
            self.references_of(self.engine), {"table": "accounts", "column": "id"}
        )


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

    def test_create_table_with_references(self):
        status, _ = self.call("POST", "/v1/tables", {
            "name": "accounts", "columns": ACCOUNTS, "primary_key": "id",
        })
        self.assertEqual(status, 201)
        status, body = self.call("POST", "/v1/tables", {
            "name": "orders", "columns": ORDERS, "primary_key": "id",
        })
        self.assertEqual(status, 201)
        by_name = {col["name"]: col for col in body["columns"]}
        self.assertEqual(
            by_name["account_id"]["references"], {"table": "accounts", "column": "id"}
        )

    def test_create_table_with_bad_reference_is_400(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "orders", "columns": ORDERS, "primary_key": "id",
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_tx_foreign_key_violation_is_409(self):
        self.call("POST", "/v1/tables", {
            "name": "accounts", "columns": ACCOUNTS, "primary_key": "id",
        })
        self.call("POST", "/v1/tables", {
            "name": "orders", "columns": ORDERS, "primary_key": "id",
        })
        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "orders", "row": {"id": 1, "account_id": 9}},
        ]})
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "accounts", "row": {"id": 1, "owner": "ann"}},
            {"op": "insert", "table": "orders", "row": {"id": 1, "account_id": 1}},
        ]})
        self.assertEqual(status, 200)
        self.assertTrue(body["committed"])
        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "delete", "table": "accounts", "pk": 1},
        ]})
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        status, row = self.call("GET", "/v1/tables/accounts/rows/1")
        self.assertEqual(status, 200)

    def test_row_insert_over_http_enforces_references(self):
        self.call("POST", "/v1/tables", {
            "name": "accounts", "columns": ACCOUNTS, "primary_key": "id",
        })
        self.call("POST", "/v1/tables", {
            "name": "orders", "columns": ORDERS, "primary_key": "id",
        })
        status, body = self.call("POST", "/v1/tables/orders/rows", {
            "rows": [{"id": 1, "account_id": 9}],
        })
        self.assertEqual(status, 409)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
