"""Engine tests: transactions, MVCC snapshots, indexes, constraints, recovery."""

import os
import shutil
import tempfile
import unittest

from kvse import Engine, StorageError
from kvse.engine import ConflictError, ConstraintError

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
    {"name": "active", "type": "bool"},
]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-engine-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", COLUMNS, "id", indexes=["name", "age"])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def add(self, pk, name, age, active=True):
        return self.engine.insert("users", {"id": pk, "name": name, "age": age, "active": active})

    # ------------------------------------------------------------ definitions
    def test_definition_validation(self):
        cases = [
            ("", COLUMNS, "id", None),
            ("t2", [{"name": "id", "type": "blob"}], "id", None),
            ("t3", [{"name": "id", "type": "int"}, {"name": "id", "type": "int"}], "id", None),
            ("t4", [{"name": "id", "type": "int"}], "nope", None),
            ("t5", [{"name": "id", "type": "int"}], "id", ["nope"]),
            ("t6", [], "id", None),
            ("users", COLUMNS, "id", None),
        ]
        for name, columns, primary_key, indexes in cases:
            with self.assertRaises(StorageError):
                self.engine.create_table(name, columns, primary_key, indexes)
        self.assertEqual(self.engine.list_tables(), ["users"])
        info = self.engine.table_info("users")
        self.assertEqual(info["primary_key"], "id")
        self.assertEqual(sorted(info["indexes"]), ["age", "name"])

    def test_constraints(self):
        self.add(1, "ann", 30)
        with self.assertRaises(ConstraintError):
            self.add(1, "dup", 1)
        with self.assertRaises(ConstraintError):
            self.add(2, "ann", 2)
        with self.assertRaises(ConstraintError):
            self.engine.insert("users", {"id": 3, "age": 3})
        with self.assertRaises(ConstraintError):
            self.engine.insert("users", {"id": "x", "name": "x", "age": 3})
        with self.assertRaises(ConstraintError):
            self.engine.insert("users", {"id": 4, "name": "d", "age": 4, "active": 1})
        with self.assertRaises(StorageError):
            self.engine.insert("users", {"id": 5, "name": "e", "age": 5, "nope": 1})
        with self.assertRaises(StorageError):
            self.engine.insert("nope", {"id": 1})
        self.engine.insert("users", {"id": 6, "name": "f"})
        self.assertIsNone(self.engine.get("users", 6)["age"])
        self.assertEqual(len(self.engine.scan("users")), 2)

    # --------------------------------------------------------- transactions
    def test_commit_and_rollback(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin()
        tx.insert("users", {"id": 2, "name": "bob", "age": 20})
        tx.update("users", 1, {"age": 31})
        self.assertEqual(tx.get("users", 1)["age"], 31)  # own writes are visible
        lsn = tx.commit()
        self.assertEqual(lsn, self.engine.lsn)
        self.assertEqual(self.engine.get("users", 1)["age"], 31)

        tx = self.engine.begin()
        tx.update("users", 1, {"age": 99})
        tx.delete("users", 2)
        tx.insert("users", {"id": 3, "name": "cid", "age": 40})
        self.assertTrue(tx.rollback())
        self.assertEqual(self.engine.get("users", 1)["age"], 31)
        self.assertEqual(self.engine.get("users", 2)["name"], "bob")
        self.assertIsNone(self.engine.get("users", 3))
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1, 2])
        with self.assertRaises(StorageError):
            tx.commit()  # a finished transaction cannot commit again

    def test_uncommitted_rows_are_invisible_and_conflict(self):
        tx = self.engine.begin()
        tx.insert("users", {"id": 1, "name": "ann", "age": 30})
        self.assertIsNone(self.engine.get("users", 1))
        self.assertEqual(self.engine.scan("users"), [])
        other = self.engine.begin()
        with self.assertRaises(ConflictError):
            other.insert("users", {"id": 1, "name": "zoe", "age": 1})
        other.rollback()
        tx.commit()
        self.assertEqual(self.engine.get("users", 1)["name"], "ann")

    def test_snapshot_isolation(self):
        self.add(1, "ann", 30)
        reader = self.engine.begin()
        self.add(2, "bob", 20)
        self.engine.update("users", 1, {"age": 31})
        self.assertIsNone(reader.get("users", 2))
        self.assertEqual(reader.get("users", 1)["age"], 30)
        self.assertEqual([r["id"] for r in reader.scan("users")], [1])
        self.assertEqual(reader.index_get("users", "name", "bob"), [])
        self.assertEqual([r["id"] for r in reader.index_range("users", "age", 0, 99)], [1])
        reader.rollback()
        self.assertEqual(self.engine.get("users", 1)["age"], 31)

    def test_write_write_conflict(self):
        self.add(1, "ann", 30)
        older = self.engine.begin()
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ConflictError):
            older.update("users", 1, {"age": 32})
        with self.assertRaises(ConflictError):
            older.delete("users", 1)
        older.rollback()
        self.assertEqual(self.engine.get("users", 1)["age"], 31)

    def test_pinned_snapshot_rejects_a_newer_row(self):
        self.add(1, "ann", 30)
        old_txid = self.engine.audit()[-1]["txid"]
        self.add(2, "bob", 20)
        tx = self.engine.begin(snapshot=old_txid)
        self.assertIsNone(tx.get("users", 2))
        with self.assertRaises(ConflictError):
            tx.update("users", 2, {"age": 21})
        tx.rollback()
        with self.assertRaises(StorageError):
            self.engine.begin(snapshot=9999)

    # --------------------------------------------------------------- indexes
    def test_index_maintenance(self):
        self.add(1, "ann", 30)
        self.add(2, "bob", 20)
        self.add(3, "cid", 40)
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 25, 45)], [1, 3])
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", None, 25)], [2])
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 25, None)], [1, 3])
        self.assertEqual(self.engine.index_get("users", "name", "bob")[0]["id"], 2)
        self.assertEqual(self.engine.index_get("users", "name", "nobody"), [])

        self.engine.update("users", 1, {"age": 10})
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 25, 45)], [3])
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 0, 15)], [1])
        self.engine.delete("users", 3)
        self.assertEqual(self.engine.index_range("users", "age", 25, 45), [])
        self.assertEqual(self.engine.index_get("users", "name", "cid"), [])
        self.assertTrue(self.engine.verify()["ok"])

        with self.assertRaises(StorageError):
            self.engine.index_range("users", "active", 0, 1)
        with self.assertRaises(StorageError):
            self.engine.create_index("users", "name")  # already indexed
        self.assertEqual(self.engine.create_index("users", "active"), "users_active_idx")

    # ----------------------------------------------------------------- query
    def test_query_subset_and_access_paths(self):
        self.add(1, "ann", 30)
        self.add(2, "bob", 20)
        self.add(3, "cid", 40)
        self.assertEqual(
            self.engine.explain("users", where=[("age", ">=", 25)]),
            {"access": "index", "index": "users_age_idx"},
        )
        self.assertEqual(
            self.engine.explain("users", where=[("active", "=", True)]),
            {"access": "scan", "index": None},
        )
        self.assertEqual(
            self.engine.explain("users", where=[("name", "=", "bob")]),
            {"access": "index", "index": "users_name_idx"},
        )

        result = self.engine.query("users", where=[("age", ">=", 25)])
        self.assertEqual(result["access"], "index")
        self.assertEqual([r["id"] for r in result["rows"]], [1, 3])

        result = self.engine.query("users", where=[("age", "<", 30)], columns=["id", "age"])
        self.assertEqual(result["rows"], [{"id": 2, "age": 20}])

        result = self.engine.query("users", where=[("name", "!=", "ann")], order_by="id", limit=1)
        self.assertEqual(result["access"], "scan")
        self.assertEqual(result["rows"], [{"id": 2, "name": "bob", "age": 20, "active": True}])

        result = self.engine.query("users", where=[("id", "=", 3)])
        self.assertEqual([r["name"] for r in result["rows"]], ["cid"])

        self.assertEqual(
            self.engine.explain("users", where=[("age", ">", 1)], index_hint="users_age_idx"),
            {"access": "index", "index": "users_age_idx"},
        )
        self.assertEqual(
            self.engine.explain("users", where=[("active", "=", True)], index_hint="age"),
            {"access": "scan", "index": None},
        )
        for bad in ({"where": [("nope", "=", 1)]}, {"where": [("age", "LIKE", 1)]},
                    {"where": [("age", "=", "x")]}, {"order_by": "nope"},
                    {"columns": ["nope"]}, {"index_hint": "nope"}):
            with self.assertRaises(StorageError):
                self.engine.query("users", **bad)

    # -------------------------------------------------------------- recovery
    def test_recovery_after_a_simulated_crash(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin()
        tx.insert("users", {"id": 2, "name": "bob", "age": 20})
        tx.commit()
        before = self.engine.scan("users")
        lsn = self.engine.lsn
        with open(self.engine.pager.data_path, "wb"):
            pass  # every page is lost, only the WAL survives
        report = self.engine.reopen()
        self.assertEqual(report["tables"], ["users"])
        self.assertEqual(self.engine.scan("users"), before)
        self.assertEqual(self.engine.get("users", 2)["name"], "bob")
        self.assertEqual(self.engine.lsn, lsn)
        self.assertTrue(self.engine.verify()["crc_ok"])
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 0, 99)], [2, 1])

        with open(self.engine.pager.wal_path, "ab") as handle:
            handle.write(b'{"lsn": 99, "txid": 99, "page_id": 1, "payl')
        self.engine.reopen()
        self.assertEqual(self.engine.scan("users"), before)

    def test_verify_and_checkpoint(self):
        self.add(1, "ann", 30)
        report = self.engine.verify()
        self.assertEqual(sorted(report), ["crc_ok", "ok", "pages", "wal_records"])
        self.assertTrue(report["ok"])
        self.assertGreater(report["pages"], 0)
        self.assertGreater(report["wal_records"], 0)
        info = self.engine.pager.checkpoint()
        self.assertGreater(info["wal_records_compacted"], 0)
        self.assertEqual(self.engine.verify()["wal_records"], 0)
        self.engine.reopen()
        self.assertEqual(self.engine.get("users", 1)["age"], 30)

    # ---------------------------------------------------------------- backup
    def test_backup_and_point_in_time_restore(self):
        self.add(1, "ann", 30)
        lsn1 = self.engine.lsn
        self.add(2, "bob", 20)
        lsn2 = self.engine.lsn
        self.engine.update("users", 1, {"age": 31})
        lsn3 = self.engine.lsn
        self.assertTrue(lsn1 < lsn2 < lsn3)

        backup_dir = os.path.join(self.root, "backup")
        manifest = self.engine.backup(backup_dir)
        self.assertEqual(manifest["format"], "kvse-backup-1")
        self.assertEqual(manifest["lsn"], lsn3)
        self.assertTrue(os.path.exists(os.path.join(backup_dir, "manifest.json")))

        restored = self.engine.restore(backup_dir, to_lsn=lsn2)
        self.assertEqual(restored["restored_lsn"], lsn2)
        rows = self.engine.scan("users")
        self.assertEqual([(r["id"], r["age"]) for r in rows], [(1, 30), (2, 20)])

        restored = self.engine.restore(backup_dir, to_lsn=lsn1)
        self.assertEqual(restored["restored_lsn"], lsn1)
        self.assertEqual([r["id"] for r in self.engine.scan("users")], [1])

        restored = self.engine.restore(backup_dir)
        self.assertEqual(restored["restored_lsn"], lsn3)
        self.assertEqual({r["id"]: r["age"] for r in self.engine.scan("users")}, {1: 31, 2: 20})
        self.assertTrue(self.engine.verify()["ok"])
        with self.assertRaises(StorageError):
            self.engine.restore(backup_dir, to_lsn=lsn3 + 100)
        with self.assertRaises(StorageError):
            self.engine.restore(os.path.join(self.root, "missing"))

    # ------------------------------------------------------------- savepoints
    def test_savepoint_rolls_back_partial_writes(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin()
        tx.insert("users", {"id": 2, "name": "bob", "age": 20})
        self.assertTrue(tx.savepoint("sp1"))
        tx.update("users", 1, {"age": 31})
        tx.update("users", 1, {"age": 32})
        tx.insert("users", {"id": 3, "name": "cid", "age": 40})
        tx.delete("users", 2)
        self.assertTrue(tx.rollback_to("sp1"))
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.assertEqual(tx.get("users", 2)["name"], "bob")
        self.assertIsNone(tx.get("users", 3))
        self.assertEqual([r["id"] for r in tx.scan("users")], [1, 2])
        self.assertEqual(tx.index_get("users", "name", "cid"), [])
        self.assertEqual([r["id"] for r in tx.index_range("users", "age", 0, 99)], [2, 1])
        # primary key and unique slots occupied after the savepoint are free again
        tx.insert("users", {"id": 3, "name": "cid", "age": 41})
        tx.commit()
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        self.assertEqual(self.engine.get("users", 3)["age"], 41)
        self.assertTrue(self.engine.verify()["ok"])

    def test_savepoint_name_validation_and_unknown_names(self):
        tx = self.engine.begin()
        for bad in (None, "", "   ", 5, b"x", ["a"]):
            with self.assertRaises(StorageError):
                tx.savepoint(bad)
            with self.assertRaises(StorageError):
                tx.rollback_to(bad)
            with self.assertRaises(StorageError):
                tx.release_savepoint(bad)
        self.assertEqual(tx.state, "active")
        self.assertTrue(tx.savepoint("a"))
        with self.assertRaises(StorageError):
            tx.savepoint("a")  # duplicate while still active
        self.assertTrue(tx.savepoint("A"))  # names are case sensitive
        with self.assertRaises(StorageError):
            tx.rollback_to("nope")
        with self.assertRaises(StorageError):
            tx.release_savepoint("nope")
        tx.rollback()

    def test_nested_savepoints_invalidate_and_release(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin()
        tx.savepoint("s1")
        tx.update("users", 1, {"age": 31})
        tx.savepoint("s2")
        tx.update("users", 1, {"age": 32})
        tx.savepoint("s3")
        tx.update("users", 1, {"age": 33})
        self.assertTrue(tx.rollback_to("s2"))
        self.assertEqual(tx.get("users", 1)["age"], 31)
        with self.assertRaises(StorageError):
            tx.rollback_to("s3")  # savepoints after the target are gone
        self.assertTrue(tx.savepoint("s3"))  # an invalidated name can be reused
        tx.update("users", 1, {"age": 34})
        self.assertTrue(tx.rollback_to("s2"))  # the target itself stays valid
        self.assertEqual(tx.get("users", 1)["age"], 31)
        self.assertTrue(tx.release_savepoint("s1"))  # drops s1 and everything after
        with self.assertRaises(StorageError):
            tx.rollback_to("s2")
        self.assertEqual(tx.get("users", 1)["age"], 31)  # release never undoes writes
        tx.commit()
        self.assertEqual(self.engine.get("users", 1)["age"], 31)

    def test_savepoints_leave_lsn_audit_and_finished_state_untouched(self):
        self.add(1, "ann", 30)
        lsn = self.engine.lsn
        audit_len = len(self.engine.audit())
        tx = self.engine.begin()
        tx.savepoint("s1")
        tx.insert("users", {"id": 2, "name": "bob", "age": 20})
        tx.savepoint("s2")
        tx.delete("users", 2)
        tx.rollback_to("s2")
        tx.release_savepoint("s1")
        self.assertEqual(self.engine.lsn, lsn)  # savepoints never advance the lsn
        self.assertEqual(len(self.engine.audit()), audit_len)
        tx.commit()
        entries = self.engine.audit()
        self.assertEqual(len(entries), audit_len + 1)
        self.assertEqual([op["op"] for op in entries[-1]["ops"]], ["insert"])
        for call in (lambda: tx.savepoint("x"), lambda: tx.rollback_to("x"),
                     lambda: tx.release_savepoint("x")):
            with self.assertRaises(StorageError):
                call()  # the transaction is finished

    def test_transaction_batch_accepts_savepoint_ops(self):
        self.add(1, "ann", 30)
        result = self.engine.transaction([
            {"op": "savepoint", "name": "sp"},
            {"op": "insert", "table": "users", "row": {"id": 2, "name": "bob", "age": 20}},
            {"op": "rollback_to", "name": "sp"},
            {"op": "insert", "table": "users", "row": {"id": 3, "name": "cid", "age": 40}},
            {"op": "release_savepoint", "name": "sp"},
        ])
        self.assertTrue(result["committed"])
        self.assertIsNone(self.engine.get("users", 2))
        self.assertEqual(self.engine.get("users", 3)["name"], "cid")
        self.assertEqual([op["op"] for op in self.engine.audit()[-1]["ops"]], ["insert"])

        with self.assertRaises(StorageError):
            self.engine.transaction([
                {"op": "insert", "table": "users", "row": {"id": 9, "name": "zed", "age": 1}},
                {"op": "rollback_to", "name": "missing"},
            ])
        self.assertIsNone(self.engine.get("users", 9))  # the whole batch rolled back

    def test_serializable_reads_survive_a_savepoint_rollback(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin(isolation="serializable")
        self.assertEqual(tx.get("users", 1)["age"], 30)
        tx.savepoint("sp")
        tx.update("users", 1, {"age": 31})
        tx.rollback_to("sp")
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.engine.update("users", 1, {"age": 99})  # commits after tx started
        with self.assertRaises(ConflictError):
            tx.commit()  # the read predicate recorded before the savepoint still protects
        self.assertEqual(tx.state, "rolled_back")
        self.assertEqual(self.engine.get("users", 1)["age"], 99)

    # ----------------------------------------------------------- audit/readonly
    def test_audit_and_read_only_view(self):
        self.add(1, "ann", 30)
        tx = self.engine.begin()
        tx.update("users", 1, {"age": 32})
        tx.commit()
        entries = self.engine.audit()
        self.assertEqual([e["lsn"] for e in entries], sorted(e["lsn"] for e in entries))
        self.assertEqual([e["txid"] for e in entries], sorted(e["txid"] for e in entries))
        last = entries[-1]
        self.assertEqual(last["ops"][0]["op"], "update")
        self.assertEqual(last["at"], 1000)
        self.assertLess(last["lsn"], self.engine.lsn)
        self.assertEqual(self.engine.audit(limit=1), [last])
        self.assertEqual(self.engine.audit(limit=0), [])

        view = self.engine.readonly_view()
        self.assertEqual(view.get("users", 1)["age"], 32)
        self.assertEqual(len(view.scan("users")), 1)
        self.assertEqual(view.audit(limit=1), [last])
        writes = [
            lambda: view.insert("users", {"id": 9, "name": "zed", "age": 1}),
            lambda: view.update("users", 1, {"age": 1}),
            lambda: view.delete("users", 1),
            lambda: view.create_table("t9", COLUMNS, "id"),
            lambda: view.create_index("users", "id"),
            lambda: view.begin(),
            lambda: view.transaction([]),
            lambda: view.restore("nowhere"),
        ]
        for call in writes:
            with self.assertRaises(StorageError):
                call()
        self.assertEqual(self.engine.get("users", 1)["age"], 32)


if __name__ == "__main__":
    unittest.main()
