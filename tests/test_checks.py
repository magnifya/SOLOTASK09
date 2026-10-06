"""CHECK constraint tests: definitions, runtime checks, atomicity, recovery."""

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from kvse import Engine, StorageError
from kvse.engine import ConstraintError
from kvse.http_app import create_server
from kvse.replica import Replica

from test_foreign_keys import rewrite_state

USERS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "age", "type": "int"},
    {"name": "nick", "type": "text"},
]

CHECKS = [
    {"name": "adult", "predicates": [["age", ">=", 18], ["age", "<", 150]]},
    {"name": "nick_ok", "predicates": [["nick", "!=", "bad"]]},
]


class CheckDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-def-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def checks_of(self, engine, table="users"):
        return engine.table_info(table)["checks"]

    def test_valid_checks_are_normalized_and_returned(self):
        info = self.engine.create_table("users", USERS, "id", checks=[
            {"name": "adult", "predicates": [["age", ">=", 18], ("age", "<", 150)]},
            {"name": "nick_ok", "predicates": [{"column": "nick", "op": "!=", "value": "bad"}]},
        ])
        self.assertEqual(info["checks"], CHECKS)
        self.assertEqual(self.checks_of(self.engine), CHECKS)
        # Mutating the returned object must not touch the catalog.
        info["checks"][0]["predicates"][0][2] = 1
        self.assertEqual(self.checks_of(self.engine), CHECKS)

    def test_missing_checks_means_no_constraints(self):
        info = self.engine.create_table("users", USERS, "id")
        self.assertEqual(info["checks"], [])
        self.engine.insert("users", {"id": 1, "age": 3})
        self.assertEqual(self.engine.get("users", 1)["age"], 3)

    def test_null_constant_is_allowed(self):
        self.engine.create_table(
            "users", USERS, "id",
            checks=[{"name": "no_nick", "predicates": [["nick", "=", None]]}],
        )
        self.engine.insert("users", {"id": 1, "nick": None})
        with self.assertRaises(ConstraintError):
            self.engine.insert("users", {"id": 2, "nick": "amy"})

    def test_invalid_definitions_raise_storage_error(self):
        bad = [
            "not-a-list",
            ["not-an-object"],
            [{"predicates": [["age", ">", 0]]}],
            [{"name": "", "predicates": [["age", ">", 0]]}],
            [{"name": "  ", "predicates": [["age", ">", 0]]}],
            [{"name": "c"}],
            [{"name": "c", "predicates": []}],
            [{"name": "c", "predicates": "age>0"}],
            [{"name": "c", "predicates": ["age>0"]}],
            [{"name": "c", "predicates": [["age", ">", 0, 1]]}],
            [{"name": "c", "predicates": [["age", "=~", 0]]}],
            [{"name": "c", "predicates": [["ghost", ">", 0]]}],
            [{"name": "c", "predicates": [["age", ">", "old"]]}],
            [{"name": "c", "predicates": [["age", ">", True]]}],
            [{"name": "c", "predicates": [["nick", "=", 1]]}],
            [
                {"name": "c", "predicates": [["age", ">", 0]]},
                {"name": "c", "predicates": [["age", "<", 9]]},
            ],
        ]
        for checks in bad:
            with self.assertRaises(StorageError, msg=repr(checks)) as ctx:
                self.engine.create_table("users", USERS, "id", checks=checks)
            self.assertNotIsInstance(ctx.exception, ConstraintError)
            self.assertFalse(self.engine.has_table("users"))

    def test_failed_create_writes_nothing(self):
        pages = self.engine.pager.page_count()
        records = self.engine.pager.wal_records()
        audits = len(self.engine.audit())
        with self.assertRaises(StorageError):
            self.engine.create_table(
                "users", USERS, "id",
                checks=[{"name": "c", "predicates": [["ghost", ">", 0]]}],
            )
        self.assertEqual(self.engine.pager.page_count(), pages)
        self.assertEqual(self.engine.pager.wal_records(), records)
        self.assertEqual(len(self.engine.audit()), audits)


class CheckRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-run-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", USERS, "id", checks=CHECKS)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_insert_enforces_checks(self):
        self.engine.insert("users", {"id": 1, "age": 30, "nick": "amy"})
        for row in (
            {"id": 2, "age": 10},
            {"id": 3, "age": 150},
            {"id": 4, "age": None},
            {"id": 5, "age": 40, "nick": "bad"},
        ):
            with self.assertRaises(ConstraintError, msg=repr(row)):
                self.engine.insert("users", row)
            self.assertIsNone(self.engine.get("users", row["id"]))
        # A null nick passes "!=" with a text constant.
        self.engine.insert("users", {"id": 6, "age": 20, "nick": None})

    def test_violation_names_table_and_constraint(self):
        with self.assertRaises(ConstraintError) as ctx:
            self.engine.insert("users", {"id": 2, "age": 10})
        self.assertIn("users", str(ctx.exception))
        self.assertIn("adult", str(ctx.exception))
        with self.assertRaises(ConstraintError) as ctx:
            self.engine.insert("users", {"id": 3, "age": 40, "nick": "bad"})
        self.assertIn("nick_ok", str(ctx.exception))

    def test_failed_op_keeps_transaction_usable(self):
        tx = self.engine.begin()
        tx.insert("users", {"id": 1, "age": 25})
        with self.assertRaises(ConstraintError):
            tx.insert("users", {"id": 2, "age": 1})
        self.assertEqual(tx.state, "active")
        tx.update("users", 1, {"age": 26})
        tx.commit()
        self.assertIsNone(self.engine.get("users", 2))
        self.assertEqual(self.engine.get("users", 1)["age"], 26)

    def test_update_enforces_checks(self):
        self.engine.insert("users", {"id": 1, "age": 30})
        with self.assertRaises(ConstraintError):
            self.engine.update("users", 1, {"age": 3})
        with self.assertRaises(ConstraintError):
            self.engine.update("users", 1, {"nick": "bad"})
        self.assertEqual(self.engine.get("users", 1), {"id": 1, "age": 30, "nick": None})

    def test_delete_is_not_checked(self):
        self.engine.insert("users", {"id": 1, "age": 30})
        self.engine.delete("users", 1)
        self.assertIsNone(self.engine.get("users", 1))

    def test_savepoint_rollback_cannot_bypass_checks(self):
        tx = self.engine.begin()
        tx.insert("users", {"id": 1, "age": 33})
        tx.savepoint("s1")
        with self.assertRaises(ConstraintError):
            tx.insert("users", {"id": 2, "age": 1})
        tx.rollback_to("s1")
        tx.insert("users", {"id": 3, "age": 44})
        tx.commit()
        self.assertIsNone(self.engine.get("users", 2))
        self.assertIsNotNone(self.engine.get("users", 1))
        self.assertIsNotNone(self.engine.get("users", 3))

    def test_commit_time_revalidation_rolls_everything_back(self):
        self.engine.insert("users", {"id": 1, "age": 30})
        lsn = self.engine.lsn
        records = self.engine.pager.wal_records()
        audits = len(self.engine.audit())
        tx = self.engine.begin()
        tx.insert("users", {"id": 2, "age": 40})
        tx.update("users", 1, {"age": 50})
        # A catalog change landing between the writes and the commit.
        self.engine.tables["users"]["checks"] = [
            {"name": "adult", "predicates": [["age", ">", 100]]},
        ]
        with self.assertRaises(ConstraintError):
            tx.commit()
        self.assertEqual(tx.state, "rolled_back")
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.pager.wal_records(), records)
        self.assertEqual(len(self.engine.audit()), audits)
        self.assertIsNone(self.engine.get("users", 2))
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        with self.assertRaises(StorageError):
            tx.insert("users", {"id": 3, "age": 40})

    def test_batch_transaction_rolls_back_on_violation(self):
        lsn = self.engine.lsn
        with self.assertRaises(ConstraintError):
            self.engine.transaction([
                {"op": "insert", "table": "users", "row": {"id": 1, "age": 30}},
                {"op": "insert", "table": "users", "row": {"id": 2, "age": 2}},
            ])
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.scan("users"), [])

    def test_queries_and_indexes_are_unaffected(self):
        self.engine.create_index("users", "age")
        self.engine.insert("users", {"id": 1, "age": 30})
        self.engine.insert("users", {"id": 2, "age": 40})
        rows = self.engine.query("users", where=[["age", ">=", 35]])["rows"]
        self.assertEqual([row["id"] for row in rows], [2])
        self.assertEqual([row["id"] for row in self.engine.index_range("users", "age", 0, 35)], [1])
        self.assertTrue(self.engine.verify()["ok"])


class CheckPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-check-persist-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", USERS, "id", checks=CHECKS)
        self.engine.insert("users", {"id": 1, "age": 30, "nick": "amy"})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def checks_of(self, engine, table="users"):
        return engine.table_info(table)["checks"]

    def test_reopen_keeps_definitions_and_enforcement(self):
        self.engine.reopen()
        self.assertEqual(self.checks_of(self.engine), CHECKS)
        with self.assertRaises(ConstraintError):
            self.engine.insert("users", {"id": 2, "age": 3})

    def test_backup_and_restore_keep_checks(self):
        backup = tempfile.mkdtemp(prefix="kvse-check-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)
        other = tempfile.mkdtemp(prefix="kvse-check-other-")
        self.addCleanup(shutil.rmtree, other, True)
        restored = Engine(other, now_ms=lambda: 1000)
        restored.restore(backup)
        self.assertEqual(self.checks_of(restored), CHECKS)
        with self.assertRaises(ConstraintError):
            restored.insert("users", {"id": 2, "age": 3})

    def test_replica_sync_keeps_checks(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-check-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        self.assertEqual(self.checks_of(replica), CHECKS)
        self.engine.insert("users", {"id": 2, "age": 40})
        replica.sync()
        self.assertEqual(replica.get("users", 2)["age"], 40)
        session = replica.read_session()
        self.assertEqual(self.checks_of(session), CHECKS)
        session.close()

    def test_image_without_checks_opens_without_migration(self):
        def strip(state):
            for table in state["tables"].values():
                table.pop("checks", None)

        rewrite_state(self.root, strip)
        report = self.engine.reopen()
        self.assertIn("users", report["tables"])
        self.assertEqual(self.checks_of(self.engine), [])
        self.engine.insert("users", {"id": 2, "age": 3})
        self.assertEqual(self.engine.get("users", 2)["age"], 3)

    def test_violating_loaded_state_is_rejected(self):
        def violate(state):
            state["tables"]["users"]["rows"].append([9, {"id": 9, "age": 3, "nick": None}])

        rewrite_state(self.root, violate)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_malformed_loaded_definition_is_rejected(self):
        def corrupt(state):
            state["tables"]["users"]["checks"] = [
                {"name": "c", "predicates": [["ghost", ">", 0]]},
            ]

        rewrite_state(self.root, corrupt)
        with self.assertRaises(StorageError):
            self.engine.reopen()

    def test_sync_of_violating_source_keeps_replica_state(self):
        replica_dir = tempfile.mkdtemp(prefix="kvse-check-replica-")
        self.addCleanup(shutil.rmtree, replica_dir, True)
        replica = Replica.create(self.engine, replica_dir)
        applied = replica.applied_lsn

        def violate(state):
            state["tables"]["users"]["rows"].append([9, {"id": 9, "age": 3, "nick": None}])

        rewrite_state(
            self.root, violate,  # corrupt the primary's checkpointed image
            meta_mutate=lambda meta: meta.update(lsn=meta["lsn"] + 10),
        )
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("users", 1)["age"], 30)
        self.assertEqual(self.checks_of(replica), CHECKS)

    def test_restore_of_violating_backup_keeps_current_state(self):
        backup = tempfile.mkdtemp(prefix="kvse-check-backup-")
        self.addCleanup(shutil.rmtree, backup, True)
        self.engine.backup(backup)

        def violate(state):
            state["tables"]["users"]["rows"].append([9, {"id": 9, "age": 3, "nick": None}])

        rewrite_state(backup, violate)
        lsn = self.engine.lsn
        with self.assertRaises(StorageError):
            self.engine.restore(backup)
        self.assertEqual(self.engine.lsn, lsn)
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        self.assertEqual(self.checks_of(self.engine), CHECKS)


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

    def test_create_table_with_checks(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "users", "columns": USERS, "primary_key": "id", "checks": CHECKS,
        })
        self.assertEqual(status, 201)
        self.assertEqual(body["checks"], CHECKS)

    def test_create_table_with_bad_checks_is_400(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "users", "columns": USERS, "primary_key": "id",
            "checks": [{"name": "c", "predicates": [["ghost", ">", 0]]}],
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, body = self.call("GET", "/v1/tables")
        self.assertNotIn("users", body["tables"])

    def test_row_violation_is_409(self):
        self.call("POST", "/v1/tables", {
            "name": "users", "columns": USERS, "primary_key": "id", "checks": CHECKS,
        })
        status, body = self.call("POST", "/v1/tables/users/rows", {
            "rows": [{"id": 1, "age": 30}],
        })
        self.assertEqual(status, 201)
        status, body = self.call("POST", "/v1/tables/users/rows", {
            "rows": [{"id": 2, "age": 3}],
        })
        self.assertEqual(status, 409)
        self.assertIn("adult", body["error"])
        self.assertIn("users", body["error"])
        status, _ = self.call("GET", "/v1/tables/users/rows/2")
        self.assertEqual(status, 404)

    def test_tx_violation_is_409(self):
        self.call("POST", "/v1/tables", {
            "name": "users", "columns": USERS, "primary_key": "id", "checks": CHECKS,
        })
        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "users", "row": {"id": 1, "age": 30}},
            {"op": "insert", "table": "users", "row": {"id": 2, "age": 2}},
        ]})
        self.assertEqual(status, 409)
        self.assertIn("error", body)
        status, row = self.call("GET", "/v1/tables/users/rows/1")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
