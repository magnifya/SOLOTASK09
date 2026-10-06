"""CHECK constraint tests: definitions, runtime enforcement, atomicity, recovery."""

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

ITEMS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "label", "type": "text", "nullable": False},
    {"name": "price", "type": "int"},
    {"name": "stock", "type": "int"},
]

CHECKS = [
    {"name": "positive_price", "predicates": [["price", ">", 0]]},
    {"name": "stock_range", "predicates": [["stock", ">=", 0], ["stock", "<=", 100]]},
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


class CheckDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-def-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def create(self, checks, name="items", columns=ITEMS):
        return self.engine.create_table(name, columns, "id", checks=checks)

    def test_valid_checks_are_stored_and_returned_normalized(self):
        info = self.create([
            {"name": "positive_price", "predicates": [["price", ">", 0]]},
            {"name": "has_label", "predicates": [{"column": "label", "op": "!=", "value": ""}]},
            {"name": "in_stock", "predicates": [{"column": "stock", "value": 0}]},
        ])
        self.assertEqual(
            info["checks"],
            [
                {"name": "positive_price", "predicates": [["price", ">", 0]]},
                {"name": "has_label", "predicates": [["label", "!=", ""]]},
                {"name": "in_stock", "predicates": [["stock", "=", 0]]},
            ],
        )
        info = self.engine.table_info("items")
        self.assertEqual(len(info["checks"]), 3)
        # Mutating the returned object must not touch the catalog.
        info["checks"][0]["predicates"][0][2] = -100
        info["checks"][0]["name"] = "nope"
        again = self.engine.table_info("items")
        self.assertEqual(again["checks"][0]["name"], "positive_price")
        self.assertEqual(again["checks"][0]["predicates"], [["price", ">", 0]])

    def test_missing_and_empty_checks_mean_no_constraint(self):
        self.engine.create_table("plain", ITEMS, "id")
        self.engine.create_table("empty", ITEMS, "id", checks=[])
        self.assertEqual(self.engine.table_info("plain")["checks"], [])
        self.assertEqual(self.engine.table_info("empty")["checks"], [])
        self.engine.insert("plain", {"id": 1, "label": "a", "price": -5, "stock": -1})
        self.engine.insert("empty", {"id": 1, "label": "a", "price": -5, "stock": -1})
        self.assertEqual(self.engine.get("plain", 1)["price"], -5)
        self.assertEqual(self.engine.get("empty", 1)["stock"], -1)

    def test_old_positional_call_still_works(self):
        info = self.engine.create_table("items", ITEMS, "id", ["price"])
        self.assertEqual(info["checks"], [])
        self.assertEqual(sorted(info["indexes"]), ["price"])

    def test_invalid_check_definitions(self):
        cases = [
            "not-a-list",
            [{"name": "c1"}],                                  # no predicates
            [{"predicates": [["price", ">", 0]]}],             # no name
            [{"name": "", "predicates": [["price", ">", 0]]}],
            [{"name": "  ", "predicates": [["price", ">", 0]]}],
            [{"name": 7, "predicates": [["price", ">", 0]]}],
            [{"name": "c1", "predicates": []}],                # no predicate
            [{"name": "c1", "predicates": "price > 0"}],
            [{"name": "c1", "predicates": ["price>0"]}],
            [{"name": "c1", "predicates": [["price", ">", 0, 1]]}],
            [{"name": "c1", "predicates": [["price", "like", 0]]}],
            [{"name": "c1", "predicates": [["nope", ">", 0]]}],
            [{"name": "c1", "predicates": [["price", ">", "x"]]}],
            [{"name": "c1", "predicates": [["price", ">", True]]}],
            [{"name": "c1", "predicates": [["label", "=", 3]]}],
            [
                {"name": "c1", "predicates": [["price", ">", 0]]},
                {"name": "c1", "predicates": [["price", "<", 9]]},
            ],                                                 # duplicate name
            ["c1"],                                            # not an object
        ]
        for checks in cases:
            with self.assertRaises(StorageError, msg=repr(checks)):
                self.create(checks)
        self.assertEqual(self.engine.list_tables(), [])

    def test_failed_create_leaves_no_trace(self):
        with self.assertRaises(StorageError):
            self.create([{"name": "c1", "predicates": [["nope", ">", 0]]}])
        self.assertEqual(self.engine.list_tables(), [])
        self.assertEqual(self.engine.audit(), [])
        self.assertEqual(self.engine.pager.page_count(), 0)
        with open(os.path.join(self.root, "wal.log"), "rb") as fh:
            self.assertEqual(fh.read(), b"")

    def test_null_constant_is_allowed_in_definition(self):
        info = self.create([{"name": "no_price", "predicates": [["price", "=", None]]}])
        self.assertEqual(info["checks"][0]["predicates"], [["price", "=", None]])


class CheckEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-run-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("items", ITEMS, "id", indexes=["price"], checks=CHECKS)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def add(self, pk, price, stock, label="x"):
        return self.engine.insert(
            "items", {"id": pk, "label": label, "price": price, "stock": stock}
        )

    def test_insert_violation_names_table_and_constraint(self):
        with self.assertRaises(ConstraintError) as ctx:
            self.add(1, -1, 5)
        self.assertIn("items", str(ctx.exception))
        self.assertIn("positive_price", str(ctx.exception))
        with self.assertRaises(ConstraintError) as ctx:
            self.add(2, 10, 101)
        self.assertIn("stock_range", str(ctx.exception))
        self.assertEqual(self.engine.scan("items"), [])

    def test_insert_and_update_within_bounds(self):
        self.add(1, 10, 5)
        updated = self.engine.update("items", 1, {"price": 20, "stock": 100})
        self.assertEqual(updated["price"], 20)
        self.assertEqual(self.engine.get("items", 1)["stock"], 100)

    def test_update_violation_is_rejected(self):
        self.add(1, 10, 5)
        with self.assertRaises(ConstraintError):
            self.engine.update("items", 1, {"price": 0})
        with self.assertRaises(ConstraintError):
            self.engine.update("items", 1, {"stock": -1})
        self.assertEqual(self.engine.get("items", 1)["price"], 10)
        self.assertEqual(self.engine.get("items", 1)["stock"], 5)

    def test_predicates_of_one_check_are_anded(self):
        self.add(1, 10, 0)      # lower bound ok
        self.add(2, 10, 100)    # upper bound ok
        with self.assertRaises(ConstraintError):
            self.add(3, 10, 101)
        self.assertEqual(len(self.engine.scan("items")), 2)

    def test_null_semantics_of_comparisons(self):
        # A range comparison is false when either side is null.
        with self.assertRaises(ConstraintError):
            self.add(1, None, 5)
        # "!=" compares directly, so null satisfies it.
        self.engine.create_table(
            "tags",
            [
                {"name": "id", "type": "int", "nullable": False},
                {"name": "code", "type": "int"},
            ],
            "id",
            checks=[{"name": "not_five", "predicates": [["code", "!=", 5]]}],
        )
        self.engine.insert("tags", {"id": 1, "code": None})
        with self.assertRaises(ConstraintError):
            self.engine.insert("tags", {"id": 2, "code": 5})
        # "=" compares directly too: null never equals a constant.
        with self.assertRaises(ConstraintError):
            self.add(2, None, 5)

    def test_transaction_stays_usable_after_violation(self):
        tx = self.engine.begin()
        tx.insert("items", {"id": 1, "label": "a", "price": 10, "stock": 1})
        with self.assertRaises(ConstraintError):
            tx.insert("items", {"id": 2, "label": "b", "price": -1, "stock": 1})
        tx.insert("items", {"id": 3, "label": "c", "price": 30, "stock": 3})
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.scan("items")], [1, 3])

    def test_failed_single_statement_leaves_no_trace(self):
        lsn = self.engine.lsn
        audit = len(self.engine.audit())
        with self.assertRaises(ConstraintError):
            self.add(1, -1, 5)
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(len(self.engine.audit()), audit)
        self.assertEqual(self.engine.scan("items"), [])

    def test_batch_transaction_is_atomic(self):
        lsn = self.engine.lsn
        ops = [
            {"op": "insert", "table": "items",
             "row": {"id": 1, "label": "a", "price": 10, "stock": 1}},
            {"op": "insert", "table": "items",
             "row": {"id": 2, "label": "b", "price": -2, "stock": 1}},
        ]
        with self.assertRaises(ConstraintError):
            self.engine.transaction(ops)
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.scan("items"), [])

    def test_savepoint_rollback_does_not_bypass_checks(self):
        tx = self.engine.begin()
        tx.insert("items", {"id": 1, "label": "a", "price": 10, "stock": 1})
        tx.savepoint("sp")
        with self.assertRaises(ConstraintError):
            tx.insert("items", {"id": 2, "label": "b", "price": -1, "stock": 1})
        tx.insert("items", {"id": 2, "label": "b", "price": 2, "stock": 2})
        tx.rollback_to("sp")
        with self.assertRaises(ConstraintError):
            tx.update("items", 1, {"stock": 200})
        tx.update("items", 1, {"stock": 50})
        tx.commit()
        self.assertEqual(len(self.engine.scan("items")), 1)
        self.assertEqual(self.engine.get("items", 1)["stock"], 50)

    def test_commit_time_revalidation_rolls_back_everything(self):
        tx = self.engine.begin()
        tx.insert("items", {"id": 1, "label": "a", "price": 10, "stock": 1})
        tx.insert("items", {"id": 2, "label": "b", "price": 20, "stock": 2})
        # Tighten the constraint after the writes ran (e.g. catalog change).
        self.engine.tables["items"]["checks"].append(
            {"name": "small_price", "predicates": [["price", "<", 15]]}
        )
        lsn = self.engine.lsn
        audit = len(self.engine.audit())
        wal_records = self.engine.pager.wal_records()
        with self.assertRaises(ConstraintError) as ctx:
            tx.commit()
        self.assertIn("small_price", str(ctx.exception))
        self.assertEqual(tx.state, "rolled_back")
        with self.assertRaises(StorageError):
            tx.insert("items", {"id": 3, "label": "c", "price": 1, "stock": 1})
        self.assertEqual(self.engine.scan("items"), [])
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(len(self.engine.audit()), audit)
        self.assertEqual(self.engine.pager.wal_records(), wal_records)
        self.engine.reopen()
        self.assertEqual(self.engine.scan("items"), [])
        self.assertEqual(self.engine.lsn, lsn)

    def test_commit_time_revalidation_skips_deleted_rows(self):
        tx = self.engine.begin()
        tx.insert("items", {"id": 1, "label": "a", "price": 20, "stock": 1})
        tx.delete("items", 1)
        tx.insert("items", {"id": 2, "label": "b", "price": 10, "stock": 2})
        self.engine.tables["items"]["checks"].append(
            {"name": "small_price", "predicates": [["price", "<", 15]]}
        )
        tx.commit()  # only the deleted row violates; it does not persist
        self.assertEqual([r["id"] for r in self.engine.scan("items")], [2])

    def test_other_constraints_still_apply(self):
        self.add(1, 10, 5)
        with self.assertRaises(ConstraintError):
            self.add(1, 10, 5)                       # primary key
        with self.assertRaises(ConstraintError):
            self.engine.insert("items", {"id": 2, "label": "x", "price": 10})
        with self.assertRaises(ConstraintError):
            self.engine.insert(
                "items", {"id": 3, "label": "x", "price": "ten", "stock": 1}
            )

    def test_reads_and_query_plans_are_unchanged(self):
        self.add(1, 10, 5)
        self.add(2, 20, 50)
        result = self.engine.query("items", where=[["price", ">", 5]])
        self.assertEqual(result["access"], "index")
        self.assertEqual([r["id"] for r in result["rows"]], [1, 2])
        self.assertEqual(
            self.engine.explain("items", where=[["price", ">", 5]]),
            {"access": "index", "index": "items_price_idx"},
        )
        view = self.engine.readonly_view()
        self.assertEqual(view.get("items", 1)["price"], 10)
        self.assertEqual(len(view.table_info("items")["checks"]), 2)
        with self.assertRaises(StorageError):
            view.insert("items", {"id": 9, "label": "z", "price": 1, "stock": 1})


class CheckPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-persist-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("items", ITEMS, "id", checks=CHECKS)
        self.engine.insert("items", {"id": 1, "label": "a", "price": 10, "stock": 5})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_reopen_keeps_definitions_and_enforcement(self):
        self.engine.reopen()
        checks = self.engine.table_info("items")["checks"]
        self.assertEqual([c["name"] for c in checks], ["positive_price", "stock_range"])
        with self.assertRaises(ConstraintError):
            self.engine.insert("items", {"id": 2, "label": "b", "price": -1, "stock": 1})
        self.engine.insert("items", {"id": 2, "label": "b", "price": 2, "stock": 1})
        self.assertEqual(self.engine.get("items", 2)["price"], 2)

    def test_state_without_checks_opens_without_migration(self):
        def strip(state):
            for table in state["tables"].values():
                table.pop("checks", None)

        rewrite_state(self.root, strip)
        report = self.engine.reopen()
        self.assertEqual(report["tables"], ["items"])
        self.assertEqual(self.engine.table_info("items")["checks"], [])
        self.assertEqual(self.engine.get("items", 1)["price"], 10)
        # No constraint is synthesized for the old image.
        self.engine.insert("items", {"id": 2, "label": "b", "price": -1, "stock": -1})
        self.assertEqual(self.engine.get("items", 2)["price"], -1)

    def test_violating_loaded_state_is_rejected(self):
        def violate(state):
            state["tables"]["items"]["rows"].append(
                [9, {"id": 9, "label": "bad", "price": -10, "stock": 1}]
            )

        rewrite_state(self.root, violate)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_malformed_loaded_check_is_rejected(self):
        def corrupt(state):
            state["tables"]["items"]["checks"] = [
                {"name": "broken", "predicates": [["price", "like", 1]]}
            ]

        rewrite_state(self.root, corrupt)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_loaded_check_with_unknown_column_is_rejected(self):
        def corrupt(state):
            state["tables"]["items"]["checks"] = [
                {"name": "broken", "predicates": [["nope", ">", 1]]}
            ]

        rewrite_state(self.root, corrupt)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_backup_and_restore_keep_checks(self):
        backup = tempfile.mkdtemp(prefix="kvse-check-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)
        other = Engine(os.path.join(self.root, "copy"), now_ms=lambda: 1000)
        # Restore into the live engine and verify both definition and data.
        result = self.engine.restore(backup)
        self.assertEqual(result["tables"], ["items"])
        self.assertEqual(len(self.engine.table_info("items")["checks"]), 2)
        with self.assertRaises(ConstraintError):
            self.engine.insert("items", {"id": 2, "label": "b", "price": -1, "stock": 1})
        del other

    def test_restore_of_violating_backup_keeps_current_state(self):
        backup = tempfile.mkdtemp(prefix="kvse-check-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)

        def violate(state):
            state["tables"]["items"]["rows"].append(
                [9, {"id": 9, "label": "bad", "price": -10, "stock": 1}]
            )

        rewrite_state(backup, violate)
        lsn = self.engine.lsn
        with self.assertRaises(StorageError):
            self.engine.restore(backup)
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.get("items", 1)["price"], 10)
        self.assertEqual(len(self.engine.table_info("items")["checks"]), 2)

    def test_replica_sync_keeps_checks_and_refuses_writes(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-check-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        checks = replica.table_info("items")["checks"]
        self.assertEqual([c["name"] for c in checks], ["positive_price", "stock_range"])
        self.engine.insert("items", {"id": 2, "label": "b", "price": 20, "stock": 2})
        replica.sync()
        self.assertEqual(replica.get("items", 2)["price"], 20)
        session = replica.read_session()
        self.assertEqual(len(session.table_info("items")["checks"]), 2)
        session.close()
        with self.assertRaises(StorageError):
            replica.insert("items", {"id": 3, "label": "c", "price": 1, "stock": 1})

    def test_sync_of_violating_source_keeps_replica_state(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-check-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        applied = replica.applied_lsn

        def violate(state):
            state["tables"]["items"]["rows"].append(
                [9, {"id": 9, "label": "bad", "price": -10, "stock": 1}]
            )

        rewrite_state(
            self.root, violate,  # corrupt the primary's checkpointed image
            meta_mutate=lambda meta: meta.update(lsn=meta["lsn"] + 10),
        )
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("items", 1)["price"], 10)
        self.assertEqual(len(replica.table_info("items")["checks"]), 2)

    def test_replica_reopen_validates_checks(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-check-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        replica.reopen()
        self.assertEqual(len(replica.table_info("items")["checks"]), 2)


class CheckHttpTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-http-")
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

    def create_items(self):
        return self.call("POST", "/v1/tables", {
            "name": "items",
            "columns": ITEMS,
            "primary_key": "id",
            "checks": CHECKS,
        })

    def test_create_table_with_checks(self):
        status, body = self.create_items()
        self.assertEqual(status, 201)
        self.assertEqual(
            body["checks"],
            [
                {"name": "positive_price", "predicates": [["price", ">", 0]]},
                {"name": "stock_range",
                 "predicates": [["stock", ">=", 0], ["stock", "<=", 100]]},
            ],
        )

    def test_create_table_without_checks_still_works(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "plain", "columns": ITEMS, "primary_key": "id",
        })
        self.assertEqual(status, 201)
        self.assertEqual(body["checks"], [])

    def test_create_table_with_bad_checks_is_400(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "items",
            "columns": ITEMS,
            "primary_key": "id",
            "checks": [{"name": "c1", "predicates": [["nope", ">", 0]]}],
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, body = self.call("GET", "/v1/tables")
        self.assertEqual(body["tables"], [])

    def test_row_insert_over_http_enforces_checks(self):
        self.create_items()
        status, body = self.call("POST", "/v1/tables/items/rows", {
            "rows": [{"id": 1, "label": "a", "price": -1, "stock": 1}],
        })
        self.assertEqual(status, 409)
        self.assertIn("positive_price", body["error"])
        self.assertIn("items", body["error"])
        status, _ = self.call("GET", "/v1/tables/items/rows/1")
        self.assertEqual(status, 404)
        status, body = self.call("POST", "/v1/tables/items/rows", {
            "rows": [{"id": 1, "label": "a", "price": 10, "stock": 1}],
        })
        self.assertEqual(status, 201)

    def test_tx_check_violation_is_409(self):
        self.create_items()
        status, body = self.call("POST", "/v1/tx", {
            "ops": [
                {"op": "insert", "table": "items",
                 "row": {"id": 1, "label": "a", "price": 10, "stock": 1}},
                {"op": "insert", "table": "items",
                 "row": {"id": 2, "label": "b", "price": 20, "stock": 500}},
            ],
        })
        self.assertEqual(status, 409)
        self.assertIn("stock_range", body["error"])
        status, _ = self.call("GET", "/v1/tables/items/rows/1")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
