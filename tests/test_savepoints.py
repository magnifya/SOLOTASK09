"""Named savepoints: partial rollback inside snapshot/serializable transactions."""

import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from kvse import Engine, StorageError
from kvse.engine import ConflictError, ConstraintError
from kvse.http_app import create_server

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
    {"name": "active", "type": "bool"},
]


class SavepointTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-savepoint-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", COLUMNS, "id", indexes=["name", "age"])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def row(self, pk, name, age, active=True):
        return {"id": pk, "name": name, "age": age, "active": active}

    # --------------------------------------------------------------- basics
    def test_rollback_to_undoes_writes_after_the_point(self):
        tx = self.engine.begin()
        tx.insert("users", self.row(1, "ann", 30))
        self.assertTrue(tx.savepoint("s"))
        tx.insert("users", self.row(2, "bob", 20))
        tx.update("users", 1, {"age": 31})
        tx.delete("users", 2)
        tx.insert("users", self.row(3, "cid", 40))
        self.assertEqual(tx.get("users", 1)["age"], 31)
        self.assertTrue(tx.rollback_to("s"))
        # point reads, scans, queries and index reads see the pre-savepoint state
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.assertIsNone(tx.get("users", 2))
        self.assertIsNone(tx.get("users", 3))
        self.assertEqual([r["id"] for r in tx.scan("users")], [1])
        self.assertEqual(tx.query("users")["rows"], [tx.get("users", 1)])
        self.assertEqual(tx.index_get("users", "name", "bob"), [])
        self.assertEqual([r["id"] for r in tx.index_range("users", "age", 0, 99)], [1])
        tx.insert("users", self.row(4, "dan", 44))  # the transaction stays alive
        lsn = tx.commit()
        self.assertEqual(lsn, self.engine.lsn)
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 4])
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        self.assertTrue(self.engine.verify()["ok"])

    def test_repeated_writes_to_the_same_row_are_fully_undone(self):
        self.engine.insert("users", self.row(1, "ann", 30))
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.update("users", 1, {"age": 31})
        tx.update("users", 1, {"age": 32})
        tx.update("users", 1, {"name": "ann2"})
        tx.rollback_to("s")
        self.assertEqual(tx.get("users", 1), self.row(1, "ann", 30))
        tx.commit()
        self.assertEqual(self.engine.get("users", 1), self.row(1, "ann", 30))
        self.assertEqual(self.engine.index_get("users", "name", "ann2"), [])
        self.assertTrue(self.engine.verify()["ok"])

    def test_primary_key_and_unique_occupancy_follow_the_rollback(self):
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.insert("users", self.row(1, "ann", 30))
        tx.rollback_to("s")
        tx.insert("users", self.row(1, "ann", 30))  # freed again: no duplicate
        tx.insert("users", self.row(2, "bob", 20))
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 2])

        tx = self.engine.begin()
        tx.savepoint("s")
        tx.delete("users", 1)
        tx.rollback_to("s")
        with self.assertRaises(ConstraintError):
            tx.insert("users", self.row(1, "dup", 1))  # key occupied again
        with self.assertRaises(ConstraintError):
            tx.insert("users", self.row(9, "ann", 1))  # unique name occupied again
        tx.rollback()

    def test_savepoint_names_are_validated(self):
        tx = self.engine.begin()
        tx.insert("users", self.row(1, "ann", 30))
        for bad in (None, "", "   ", "\t\n", 5, True, ["s"], b"s"):
            with self.assertRaises(StorageError):
                tx.savepoint(bad)
            with self.assertRaises(StorageError):
                tx.rollback_to(bad)
            with self.assertRaises(StorageError):
                tx.release_savepoint(bad)
        # nothing changed: the write, the transaction and the savepoints are intact
        tx.savepoint("s")
        tx.savepoint("S")  # case-sensitive: a different name
        with self.assertRaises(StorageError):
            tx.savepoint("s")  # duplicate of a live name
        self.assertEqual([r["id"] for r in tx.scan("users")], [1])
        self.assertEqual(tx.state, "active")
        tx.rollback_to("S")
        tx.rollback_to("s")  # rolling back to "s" invalidates the later "S"
        with self.assertRaises(StorageError):
            tx.rollback_to("S")
        tx.commit()
        self.assertEqual(self.engine.get("users", 1)["name"], "ann")

    def test_unknown_names_raise_without_changing_anything(self):
        tx = self.engine.begin()
        tx.insert("users", self.row(1, "ann", 30))
        tx.savepoint("s")
        with self.assertRaises(StorageError):
            tx.rollback_to("nope")
        with self.assertRaises(StorageError):
            tx.release_savepoint("nope")
        self.assertEqual(tx.get("users", 1)["name"], "ann")
        tx.rollback_to("s")  # the real savepoint still works
        tx.commit()

    def test_savepoint_nesting_and_invalidation(self):
        tx = self.engine.begin()
        tx.savepoint("a")
        tx.insert("users", self.row(1, "ann", 30))
        tx.savepoint("b")
        tx.insert("users", self.row(2, "bob", 20))
        tx.savepoint("c")
        tx.insert("users", self.row(3, "cid", 40))
        tx.rollback_to("b")
        self.assertEqual([r["id"] for r in tx.scan("users")], [1])
        with self.assertRaises(StorageError):
            tx.rollback_to("c")  # savepoints after the target are gone
        tx.savepoint("c")  # ... so the name can be created again
        tx.insert("users", self.row(4, "dan", 44))
        tx.rollback_to("b")  # the target itself can be rolled back to again
        self.assertEqual([r["id"] for r in tx.scan("users")], [1])
        tx.rollback_to("a")  # earlier savepoints survive as well
        self.assertEqual(tx.scan("users"), [])
        tx.insert("users", self.row(5, "eve", 50))
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [5])

    def test_release_forgets_without_undoing(self):
        tx = self.engine.begin()
        tx.savepoint("a")
        tx.insert("users", self.row(1, "ann", 30))
        tx.savepoint("b")
        tx.insert("users", self.row(2, "bob", 20))
        self.assertTrue(tx.release_savepoint("b"))
        self.assertEqual([r["id"] for r in tx.scan("users")], [1, 2])  # nothing undone
        with self.assertRaises(StorageError):
            tx.rollback_to("b")  # released means forgotten
        tx.savepoint("b")  # ... and the name is free again
        tx.release_savepoint("a")  # releases "a" and everything after it
        with self.assertRaises(StorageError):
            tx.rollback_to("b")
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 2])

        tx = self.engine.begin()
        tx.savepoint("a")
        tx.insert("users", self.row(3, "cid", 40))
        tx.release_savepoint("a")
        tx.rollback()  # a full rollback still undoes every write
        self.assertIsNone(self.engine.get("users", 3))

    def test_savepoints_keep_identity_snapshot_and_isolation(self):
        tx = self.engine.begin(isolation="snapshot")
        identity = (tx.txid, tx.snapshot, tx.isolation)
        tx.savepoint("s")
        tx.insert("users", self.row(1, "ann", 30))
        tx.rollback_to("s")
        tx.release_savepoint("s")
        self.assertEqual((tx.txid, tx.snapshot, tx.isolation), identity)
        self.assertEqual(tx.state, "active")
        tx.rollback()

    def test_savepoint_methods_reject_a_finished_transaction(self):
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.commit()
        for call in (lambda: tx.savepoint("x"), lambda: tx.rollback_to("s"),
                     lambda: tx.release_savepoint("s")):
            with self.assertRaises(StorageError):
                call()
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.rollback()
        for call in (lambda: tx.savepoint("x"), lambda: tx.rollback_to("s"),
                     lambda: tx.release_savepoint("s")):
            with self.assertRaises(StorageError):
                call()

    # ------------------------------------------------------ lsn/audit/ops log
    def test_savepoints_leave_lsn_and_audit_untouched(self):
        self.engine.insert("users", self.row(1, "ann", 30))
        lsn_before, audit_before = self.engine.lsn, len(self.engine.audit())
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.update("users", 1, {"age": 31})
        tx.rollback_to("s")
        tx.release_savepoint("s")
        self.assertEqual((self.engine.lsn, len(self.engine.audit())), (lsn_before, audit_before))
        tx.update("users", 1, {"age": 32})
        tx.commit()
        entry = self.engine.audit()[-1]
        self.assertEqual(len(self.engine.audit()), audit_before + 1)
        # only the surviving business op is recorded; savepoints never appear
        self.assertEqual(entry["ops"], [{"op": "update", "table": "users", "pk": 1,
                                         "patch": {"age": 32}}])

    def test_commit_after_partial_rollback_persists_only_what_survived(self):
        tx = self.engine.begin()
        tx.insert("users", self.row(1, "ann", 30))
        tx.savepoint("s")
        tx.insert("users", self.row(2, "bob", 20))
        tx.rollback_to("s")
        tx.commit()
        entry = self.engine.audit()[-1]
        self.assertEqual([op["op"] for op in entry["ops"]], ["insert"])
        self.assertEqual(entry["ops"][0]["pk"], 1)
        self.engine.reopen()
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1])
        self.assertTrue(self.engine.verify()["ok"])

    def test_restart_and_restore_keep_only_committed_data(self):
        self.engine.insert("users", self.row(1, "ann", 30))
        tx = self.engine.begin()
        tx.savepoint("s")
        tx.insert("users", self.row(2, "bob", 20))
        tx.rollback_to("s")
        tx.insert("users", self.row(3, "cid", 40))
        tx.commit()
        backup_dir = os.path.join(self.root, "backup")
        self.engine.backup(backup_dir)
        self.engine.reopen()
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 3])
        self.engine.restore(backup_dir)
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 3])
        self.assertTrue(self.engine.verify()["ok"])

    # ------------------------------------------------------------ serializable
    def test_serializable_predicates_survive_a_partial_rollback(self):
        self.engine.insert("users", self.row(1, "ann", 30))
        tx = self.engine.begin(isolation="serializable")
        tx.savepoint("s")
        tx.get("users", 1)  # read inside the region that is rolled back
        tx.insert("users", self.row(2, "bob", 20))
        tx.rollback_to("s")
        self.assertIsNone(tx.get("users", 2))
        self.engine.update("users", 1, {"age": 31})  # hits the recorded predicate
        lsn_before = self.engine.lsn
        with self.assertRaises(ConflictError):
            tx.commit()
        self.assertEqual(self.engine.lsn, lsn_before)
        self.assertEqual(self.engine.get("users", 1)["age"], 31)
        self.assertIsNone(self.engine.get("users", 2))
        with self.assertRaises(StorageError):  # the conflict ends the transaction
            tx.savepoint("later")

    def test_serializable_savepoints_commit_when_no_conflict(self):
        self.engine.insert("users", self.row(1, "ann", 30))
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        tx.savepoint("s")
        tx.update("users", 1, {"age": 31})
        tx.rollback_to("s")
        tx.update("users", 1, {"age": 32})
        self.assertTrue(tx.commit() > 0)
        self.assertEqual(self.engine.get("users", 1)["age"], 32)

    # --------------------------------------------------------- batch transaction
    def test_engine_transaction_accepts_savepoint_ops(self):
        result = self.engine.transaction([
            {"op": "insert", "table": "users", "row": self.row(1, "ann", 30)},
            {"op": "savepoint", "name": "s"},
            {"op": "insert", "table": "users", "row": self.row(2, "bob", 20)},
            {"op": "rollback_to", "name": "s"},
            {"op": "insert", "table": "users", "row": self.row(3, "cid", 40)},
            {"op": "savepoint", "name": "t"},
            {"op": "delete", "table": "users", "pk": 1},
            {"op": "release_savepoint", "name": "t"},
        ])
        self.assertTrue(result["committed"])
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [3])
        entry = self.engine.audit()[-1]
        self.assertEqual([op["op"] for op in entry["ops"]],
                         ["insert", "insert", "delete"])

    def test_engine_transaction_rolls_back_everything_on_a_savepoint_error(self):
        for ops in (
            [{"op": "insert", "table": "users", "row": self.row(1, "ann", 30)},
             {"op": "rollback_to", "name": "missing"}],
            [{"op": "insert", "table": "users", "row": self.row(1, "ann", 30)},
             {"op": "release_savepoint", "name": "missing"}],
            [{"op": "savepoint", "name": "  "}],
            [{"op": "savepoint"}],  # no name at all
            [{"op": "savepoint", "name": "s"}, {"op": "savepoint", "name": "s"}],
        ):
            with self.assertRaises(StorageError):
                self.engine.transaction(ops)
            self.assertEqual(self.engine.scan("users"), [])  # fully rolled back
        self.assertEqual(self.engine._open, set())


class SavepointHttpTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-savepoint-http-")
        self.engine = Engine(self.root, now_ms=lambda: 7000)
        self.server = create_server(self.engine, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, _ = self.call("POST", "/v1/tables", {
            "name": "users", "columns": COLUMNS, "primary_key": "id", "indexes": ["age"],
        })
        self.assertEqual(status, 201)

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

    def test_tx_endpoint_accepts_savepoint_ops(self):
        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "users", "row": {"id": 1, "name": "ann", "age": 30}},
            {"op": "savepoint", "name": "s"},
            {"op": "insert", "table": "users", "row": {"id": 2, "name": "bob", "age": 20}},
            {"op": "rollback_to", "name": "s"},
            {"op": "insert", "table": "users", "row": {"id": 3, "name": "cid", "age": 40}},
            {"op": "release_savepoint", "name": "s"},
        ]})
        self.assertEqual(status, 200)
        self.assertTrue(body["committed"])
        self.assertGreater(body["lsn"], 0)
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/2")[0], 404)
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/3")[0], 200)
        # the audit holds only the surviving business ops, no savepoint directives
        entries = self.call("GET", "/v1/audit")[1]["entries"]
        self.assertEqual([op["op"] for op in entries[-1]["ops"]], ["insert", "insert"])

    def test_tx_endpoint_savepoint_errors(self):
        for ops, status_wanted in (
            ([{"op": "savepoint", "name": ""}], 400),
            ([{"op": "savepoint", "name": "   "}], 400),
            ([{"op": "savepoint"}], 400),
            ([{"op": "rollback_to", "name": "nope"}], 400),
            ([{"op": "release_savepoint", "name": "nope"}], 400),
            ([{"op": "savepoint", "name": "s"}, {"op": "savepoint", "name": "s"}], 400),
            ([{"op": "insert", "table": "users", "row": {"id": 1, "name": "ann", "age": 30}},
              {"op": "savepoint", "name": "s"},
              {"op": "insert", "table": "users", "row": {"id": 1, "name": "dup", "age": 1}},
              {"op": "rollback_to", "name": "s"}], 409),  # constraint error: whole tx fails
        ):
            status, body = self.call("POST", "/v1/tx", {"ops": ops})
            self.assertEqual(status, status_wanted)
            self.assertIn("error", body)
        self.assertEqual(self.call("POST", "/v1/query", {"table": "users"})[1]["rows"], [])


if __name__ == "__main__":
    unittest.main()
