"""Serializable isolation: commit-time read validation on top of snapshots."""

import os
import shutil
import tempfile
import unittest

from kvse import Engine, StorageError
from kvse.engine import ConflictError

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
    {"name": "active", "type": "bool"},
]


class SerializableTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-serializable-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", COLUMNS, "id", indexes=["name", "age"])
        self.engine.insert("users", {"id": 1, "name": "ann", "age": 30, "active": True})
        self.engine.insert("users", {"id": 2, "name": "bob", "age": 20, "active": False})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def begin(self, **kwargs):
        return self.engine.begin(isolation="serializable", **kwargs)

    # ------------------------------------------------------------- parameters
    def test_isolation_parameter_validation(self):
        tx = self.engine.begin()  # omitted keeps the snapshot semantics
        self.assertEqual(tx.isolation, "snapshot")
        tx.rollback()
        tx = self.engine.begin(isolation="snapshot")
        self.assertEqual(tx.isolation, "snapshot")
        tx.rollback()
        tx = self.engine.begin(isolation=None)
        self.assertEqual(tx.isolation, "snapshot")
        tx.rollback()

        for bad in ("serial", "SNAPSHOT", "", 5, True, b"serializable", ["snapshot"]):
            with self.assertRaises(StorageError):
                self.engine.begin(isolation=bad)
            with self.assertRaises(StorageError):
                self.engine.transaction([], isolation=bad)
        self.assertEqual(self.engine._open, set())  # no transaction was created

        snapshot = self.engine.audit()[-1]["txid"]
        with self.assertRaises(StorageError):
            self.engine.begin(snapshot=snapshot, isolation="serializable")
        with self.assertRaises(StorageError):
            self.engine.transaction([], snapshot=snapshot, isolation="serializable")
        self.assertEqual(self.engine._open, set())
        tx = self.engine.begin(snapshot=snapshot)  # snapshot alone stays legal
        tx.rollback()

    def test_serializable_reads_see_start_snapshot_and_own_writes(self):
        tx = self.begin()
        self.assertEqual(tx.get("users", 1)["age"], 30)
        tx.get("users", 2)
        self.engine.update("users", 2, {"age": 21})  # commits after tx started
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})
        self.assertEqual(tx.get("users", 1)["age"], 30)  # still the start snapshot
        self.assertIsNone(tx.get("users", 3))
        tx.update("users", 1, {"age": 32})  # own writes are visible
        self.assertEqual(tx.get("users", 1)["age"], 32)
        with self.assertRaises(ConflictError):
            tx.commit()  # ... but the newer committed write to id 2 conflicts
        self.assertEqual(self.engine.get("users", 1)["age"], 30)  # rolled back

    # ------------------------------------------------------------- get ranges
    def test_get_protects_an_existing_key(self):
        tx = self.begin()
        self.assertEqual(tx.get("users", 1)["name"], "ann")
        self.engine.update("users", 1, {"age": 31})
        lsn_before, audit_before = self.engine.lsn, len(self.engine.audit())
        with self.assertRaises(ConflictError):
            tx.commit()
        self.assertEqual((self.engine.lsn, len(self.engine.audit())), (lsn_before, audit_before))
        self.assertEqual(self.engine.get("users", 1)["age"], 31)

    def test_get_protects_a_missing_key(self):
        tx = self.begin()
        self.assertIsNone(tx.get("users", 99))
        self.engine.insert("users", {"id": 99, "name": "zed", "age": 1})
        with self.assertRaises(ConflictError):
            tx.commit()
        tx = self.begin()
        self.assertIsNone(tx.get("users", 98))
        self.engine.insert("users", {"id": 97, "name": "why", "age": 1})  # a different key
        self.assertTrue(tx.commit() > 0)

    def test_get_protects_against_delete(self):
        tx = self.begin()
        tx.get("users", 2)
        self.engine.delete("users", 2)
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_conflict_rolls_back_and_terminates_the_transaction(self):
        tx = self.begin()
        tx.get("users", 1)
        tx.insert("users", {"id": 5, "name": "eve", "age": 50})
        tx.update("users", 2, {"age": 21})
        self.engine.update("users", 1, {"age": 31})
        lsn_before, audit_before = self.engine.lsn, len(self.engine.audit())
        with self.assertRaises(ConflictError):
            tx.commit()
        # row and index changes are undone, nothing was persisted
        self.assertIsNone(self.engine.get("users", 5))
        self.assertEqual(self.engine.get("users", 2)["age"], 20)
        self.assertEqual(self.engine.index_get("users", "name", "eve"), [])
        self.assertEqual((self.engine.lsn, len(self.engine.audit())), (lsn_before, audit_before))
        self.assertTrue(self.engine.verify()["ok"])
        # the transaction is terminated: every further use raises StorageError
        for call in (lambda: tx.get("users", 1), lambda: tx.scan("users"),
                     lambda: tx.insert("users", {"id": 6, "name": "f", "age": 1}),
                     lambda: tx.update("users", 1, {"age": 1}), lambda: tx.delete("users", 1),
                     lambda: tx.query("users"), lambda: tx.commit(), lambda: tx.rollback()):
            with self.assertRaises(StorageError):
                call()

    # ------------------------------------------------------- scan and indexes
    def test_scan_protects_the_whole_table(self):
        tx = self.begin()
        self.assertEqual(len(tx.scan("users")), 2)
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_index_get_protects_the_equality_range(self):
        tx = self.begin()
        self.assertEqual(tx.index_get("users", "name", "ann")[0]["id"], 1)
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})  # outside
        self.assertTrue(tx.commit() > 0)

        tx = self.begin()
        self.assertEqual(tx.index_get("users", "age", 20)[0]["id"], 2)
        self.engine.update("users", 1, {"age": 20})  # moves into the range
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_index_range_protects_the_bounds(self):
        tx = self.begin()
        self.assertEqual([r["id"] for r in tx.index_range("users", "age", 15, 25)], [2])
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 22})  # inside
        with self.assertRaises(ConflictError):
            tx.commit()

        tx = self.begin()
        tx.index_range("users", "age", 15, 25)
        self.engine.insert("users", {"id": 4, "name": "dan", "age": 60})  # outside
        self.assertTrue(tx.commit() > 0)

        tx = self.begin()
        tx.index_range("users", "age", 15, 25)
        self.engine.update("users", 2, {"age": 60})  # leaves the range: pre-image matches
        with self.assertRaises(ConflictError):
            tx.commit()

        tx = self.begin()
        tx.index_range("users", "age", None, 25)  # open bounds work too
        self.engine.delete("users", 1)  # age 30 is outside: no conflict
        self.assertTrue(tx.commit() > 0)

        self.engine.update("users", 2, {"age": 20})
        tx = self.begin()
        tx.index_range("users", "age", None, 25)
        self.engine.delete("users", 2)  # age 20 is inside the open range
        with self.assertRaises(ConflictError):
            tx.commit()

    # ------------------------------------------------------------------ query
    def test_query_protects_the_full_predicate_range(self):
        tx = self.begin()
        result = tx.query("users", where=[("age", ">=", 25)], columns=["id"], limit=1,
                          order_by="id", index_hint="age")
        self.assertEqual(result["rows"], [{"id": 1}])
        # projection, limit, ordering and the index access path do not narrow it
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})
        with self.assertRaises(ConflictError):
            tx.commit()

        tx = self.begin()
        tx.query("users", where=[("age", ">=", 25), ("active", "=", True)])
        self.engine.insert("users", {"id": 4, "name": "dan", "age": 40, "active": False})
        self.assertTrue(tx.commit() > 0)  # only one predicate matches: no conflict

        tx = self.begin()
        tx.query("users", where=[("age", ">=", 25)])
        self.engine.insert("users", {"id": 5, "name": "eve", "age": 10})  # outside
        self.assertTrue(tx.commit() > 0)

    def test_query_without_predicates_protects_the_whole_table(self):
        tx = self.begin()
        self.assertEqual(len(tx.query("users")["rows"]), 2)
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_failed_reads_do_not_widen_the_protected_range(self):
        tx = self.begin()
        with self.assertRaises(StorageError):
            tx.get("nope", 1)
        with self.assertRaises(StorageError):
            tx.query("users", where=[("nope", "=", 1)])
        with self.assertRaises(StorageError):
            tx.index_range("users", "active", 0, 1)
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})
        self.assertTrue(tx.commit() > 0)  # nothing was protected by the failures

    # --------------------------------------------------------- conflict rules
    def test_changing_the_value_back_still_conflicts(self):
        tx = self.begin()
        tx.get("users", 1)
        writer = self.engine.begin()
        writer.update("users", 1, {"age": 31})
        writer.update("users", 1, {"age": 30})  # back to the original value
        writer.commit()
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_uncommitted_and_rolled_back_writes_do_not_conflict(self):
        tx = self.begin()
        tx.get("users", 1)
        writer = self.engine.begin()
        writer.update("users", 1, {"age": 31})
        self.assertTrue(tx.commit() > 0)  # the writer has not committed

        tx = self.begin()
        tx.get("users", 2)
        writer.delete("users", 2)
        writer.rollback()
        self.assertTrue(tx.commit() > 0)  # a rolled back writer does not conflict
        self.assertEqual(self.engine.get("users", 1)["age"], 30)

    def test_own_writes_do_not_conflict(self):
        tx = self.begin()
        tx.get("users", 1)
        tx.scan("users")
        tx.update("users", 1, {"age": 33})
        tx.insert("users", {"id": 3, "name": "cid", "age": 40})
        self.assertTrue(tx.commit() > 0)
        self.assertEqual(self.engine.get("users", 1)["age"], 33)

    def test_writer_isolation_and_start_order_do_not_matter(self):
        writer = self.engine.begin()  # a snapshot writer started *before* the reader
        tx = self.begin()
        tx.get("users", 1)
        writer.update("users", 1, {"age": 31})
        writer.commit()
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_cross_reading_serializable_transactions_cannot_both_commit(self):
        first = self.begin()
        second = self.begin()
        first.get("users", 1)
        second.get("users", 2)
        first.update("users", 2, {"age": 21})
        second.update("users", 1, {"age": 31})
        self.assertTrue(first.commit() > 0)
        with self.assertRaises(ConflictError):
            second.commit()
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        self.assertEqual(self.engine.get("users", 2)["age"], 21)

    def test_read_only_serializable_transaction(self):
        tx = self.begin()
        tx.get("users", 1)
        tx.scan("users")
        audit_before = len(self.engine.audit())
        lsn = tx.commit()  # no conflict: the usual commit return value
        self.assertEqual(lsn, self.engine.lsn)
        self.assertEqual(len(self.engine.audit()), audit_before + 1)  # and audit semantics

        tx = self.begin()
        tx.get("users", 1)
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_write_write_conflicts_keep_the_original_rules(self):
        tx = self.begin()
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ConflictError):
            tx.update("users", 1, {"age": 32})  # still detected at write time
        tx.rollback()

    def test_repeated_reads_keep_the_initial_snapshot(self):
        tx = self.begin()
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.engine.update("users", 1, {"age": 31})
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.assertEqual(tx.scan("users")[0]["age"], 30)
        with self.assertRaises(ConflictError):
            tx.commit()

    # --------------------------------------------------------------- recovery
    def test_committed_data_survives_reopen_backup_and_restore(self):
        tx = self.begin()
        tx.get("users", 1)
        tx.update("users", 1, {"age": 33})
        tx.commit()

        tx = self.begin()
        tx.get("users", 2)
        tx.insert("users", {"id": 9, "name": "ivy", "age": 90})
        self.engine.update("users", 2, {"age": 21})
        with self.assertRaises(ConflictError):
            tx.commit()

        self.engine.reopen()
        self.assertEqual(self.engine.get("users", 1)["age"], 33)
        self.assertIsNone(self.engine.get("users", 9))  # the failed transaction is invisible
        self.assertEqual(self.engine.get("users", 2)["age"], 21)

        backup_dir = os.path.join(self.root, "backup")
        self.engine.backup(backup_dir)
        self.engine.update("users", 1, {"age": 44})
        lsn = self.engine.lsn
        restored = self.engine.restore(backup_dir)
        self.assertLess(restored["restored_lsn"], lsn)
        self.assertEqual(self.engine.get("users", 1)["age"], 33)
        self.assertIsNone(self.engine.get("users", 9))
        self.assertTrue(self.engine.verify()["ok"])

        # serializable transactions still validate after a restart
        tx = self.begin()
        tx.get("users", 1)
        self.engine.update("users", 1, {"age": 55})
        with self.assertRaises(ConflictError):
            tx.commit()

    def test_batch_transaction_accepts_isolation(self):
        result = self.engine.transaction(
            [{"op": "insert", "table": "users", "row": {"id": 3, "name": "cid", "age": 40}}],
            isolation="serializable",
        )
        self.assertTrue(result["committed"])
        self.assertEqual(self.engine.get("users", 3)["age"], 40)
        result = self.engine.transaction(
            [{"op": "update", "table": "users", "pk": 3, "patch": {"age": 41}}],
            isolation="snapshot",
        )
        self.assertTrue(result["committed"])


if __name__ == "__main__":
    unittest.main()
