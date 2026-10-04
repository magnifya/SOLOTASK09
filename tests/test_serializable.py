"""Tests for the optional serializable isolation level.

Serializable transactions snapshot the committed state at start, keep that
snapshot for every read, and at commit time reject (read-write conflict) when
any other committed transaction wrote a protected row or predicate.  On
conflict the transaction is fully rolled back and terminated.
"""

import os
import shutil
import tempfile
import unittest

from kvse import Engine, StorageError
from kvse.engine import ConflictError, ReadConflictError

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False},
    {"name": "age", "type": "int"},
    {"name": "active", "type": "bool"},
]


class SerializableTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-ser-")
        self.engine = Engine(self.root, now_ms=lambda: 1000)
        self.engine.create_table("users", COLUMNS, "id", indexes=["name", "age"])
        self.engine.insert("users", {"id": 1, "name": "ann", "age": 30})
        self.engine.insert("users", {"id": 2, "name": "bob", "age": 20})
        self.engine.insert("users", {"id": 3, "name": "cid", "age": 40})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # -------------------------------------------------------------- validation
    def test_default_and_explicit_snapshot_keep_old_semantics(self):
        tx = self.engine.begin()
        self.assertEqual(tx.isolation, "snapshot")
        tx.rollback()
        tx = self.engine.begin(isolation="snapshot")
        self.assertEqual(tx.isolation, "snapshot")
        tx.rollback()

    def test_invalid_isolation_is_rejected_before_creating_a_tx(self):
        next_txid = self.engine._next_txid
        for bad in (123, None, ["snapshot"], ("serializable",), 1.0, "read-committed"):
            with self.assertRaises(StorageError):
                self.engine.begin(isolation=bad)
            with self.assertRaises(StorageError):
                self.engine.transaction(
                    [{"op": "insert", "table": "users", "row": {"id": 9, "name": "z", "age": 1}}],
                    isolation=bad,
                )
        self.assertEqual(self.engine._next_txid, next_txid)  # nothing was created
        self.assertEqual({r["id"] for r in self.engine.scan("users")}, {1, 2, 3})

    def test_serializable_rejects_a_pinned_snapshot(self):
        pinned = self.engine.audit()[-1]["txid"]
        with self.assertRaises(StorageError):
            self.engine.begin(snapshot=pinned, isolation="serializable")
        with self.assertRaises(StorageError):
            self.engine.transaction(
                [{"op": "update", "table": "users", "pk": 1, "patch": {"age": 9}}],
                snapshot=pinned,
                isolation="serializable",
            )

    # ------------------------------------------------------- snapshot semantics
    def test_reads_committed_state_at_start_and_own_writes(self):
        tx = self.engine.begin(isolation="serializable")
        self.engine.insert("users", {"id": 4, "name": "dan", "age": 50})
        self.engine.update("users", 1, {"age": 31})
        self.assertIsNone(tx.get("users", 4))
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.assertEqual([r["id"] for r in tx.scan("users")], [1, 2, 3])
        tx.rollback()

        # own writes are visible within the transaction
        tx = self.engine.begin(isolation="serializable")
        tx.update("users", 2, {"age": 77})
        tx.insert("users", {"id": 5, "name": "eve", "age": 5})
        self.assertEqual(tx.get("users", 2)["age"], 77)
        self.assertEqual(tx.get("users", 5)["name"], "eve")
        tx.rollback()

    def test_repeatable_reads_ignore_later_commits(self):
        tx = self.engine.begin(isolation="serializable")
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.engine.update("users", 1, {"age": 31})
        self.engine.delete("users", 2)
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.assertEqual(tx.get("users", 2)["name"], "bob")
        # index reads are ordered by the indexed value: 20, 30, 40
        self.assertEqual(
            [r["id"] for r in tx.index_range("users", "age", 0, 99)], [2, 1, 3]
        )
        tx.rollback()

    # ------------------------------------------------------------------ get
    def test_get_protects_present_and_missing_primary_key(self):
        # present key
        tx = self.engine.begin(isolation="serializable")
        self.assertEqual(tx.get("users", 1)["name"], "ann")
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ReadConflictError):
            tx.commit()
        self.assertEqual(tx.state, "failed")
        self.assertEqual(self.engine.get("users", 1)["age"], 31)  # other kept

        # missing key: a later insert there is a conflict
        tx = self.engine.begin(isolation="serializable")
        self.assertIsNone(tx.get("users", 99))
        self.engine.insert("users", {"id": 99, "name": "new", "age": 1})
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_get_does_not_conflict_on_unrelated_keys(self):
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        self.engine.update("users", 2, {"age": 21})
        self.assertEqual(tx.commit(), self.engine.lsn)

    # ----------------------------------------------------------------- scan
    def test_scan_protects_the_whole_table(self):
        tx = self.engine.begin(isolation="serializable")
        self.assertEqual(len(tx.scan("users")), 3)
        self.engine.insert("users", {"id": 10, "name": "zoe", "age": 1})
        with self.assertRaises(ReadConflictError):
            tx.commit()

        tx = self.engine.begin(isolation="serializable")
        tx.scan("users")
        self.engine.update("users", 3, {"age": 41})
        with self.assertRaises(ReadConflictError):
            tx.commit()

    # --------------------------------------------------------------- indexes
    def test_index_get_protects_equality_range(self):
        tx = self.engine.begin(isolation="serializable")
        tx.index_get("users", "name", "ann")
        # a new row entering the equality value conflicts (phantom)
        self.engine.insert("users", {"id": 20, "name": "ann", "age": 5})
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_index_range_protects_bounds_only(self):
        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 25, 45)  # matches ann(30), cid(40)
        # inside-range update conflicts
        self.engine.update("users", 1, {"age": 35})
        with self.assertRaises(ReadConflictError):
            tx.commit()

        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 25, 45)
        # a row leaving the range conflicts on the before image
        self.engine.update("users", 3, {"age": 99})
        with self.assertRaises(ReadConflictError):
            tx.commit()

        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 25, 45)
        # a row entering the range conflicts on the after image
        self.engine.update("users", 2, {"age": 26})
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_write_outside_index_range_is_no_conflict(self):
        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 0, 10)
        self.engine.insert("users", {"id": 30, "name": "hi", "age": 80})
        self.engine.update("users", 3, {"age": 41})
        self.assertEqual(tx.commit(), self.engine.lsn)

    def test_open_ended_index_range_protects_the_half_line(self):
        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 30, None)
        self.engine.update("users", 2, {"age": 29})  # stays below 30
        self.assertEqual(tx.commit(), self.engine.lsn)

        tx = self.engine.begin(isolation="serializable")
        tx.index_range("users", "age", 30, None)
        self.engine.update("users", 2, {"age": 30})  # enters range
        with self.assertRaises(ReadConflictError):
            tx.commit()

    # ------------------------------------------------------------------ query
    def test_query_protects_full_where_regardless_of_projection_limit_order(self):
        # limit does not shrink the protected range
        tx = self.engine.begin(isolation="serializable")
        tx.query("users", where=[("age", ">=", 25)], limit=1)
        self.engine.update("users", 3, {"age": 41})  # not among the 1 returned
        with self.assertRaises(ReadConflictError):
            tx.commit()

        # projection does not shrink it
        tx = self.engine.begin(isolation="serializable")
        tx.query("users", columns=["id"], where=[("active", "=", True)])
        self.engine.insert("users", {"id": 40, "name": "p", "age": 1, "active": True})
        with self.assertRaises(ReadConflictError):
            tx.commit()

        # index access path does not shrink the conjunction
        tx = self.engine.begin(isolation="serializable")
        tx.query("users", where=[("age", ">", 0), ("name", "=", "zz")])
        self.engine.insert("users", {"id": 41, "name": "zz", "age": 99})
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_query_without_where_protects_the_whole_table(self):
        tx = self.engine.begin(isolation="serializable")
        tx.query("users")
        self.engine.delete("users", 1)
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_failed_read_does_not_enlarge_protection(self):
        tx = self.engine.begin(isolation="serializable")
        with self.assertRaises(StorageError):
            tx.query("nope_table")
        with self.assertRaises(StorageError):
            tx.query("users", where=[("age", "=", "not-an-int")])
        tx.scan("users")  # only this protects
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ReadConflictError):
            tx.commit()

        # a query that fails on limit/columns after reading still recorded nothing
        tx = self.engine.begin(isolation="serializable")
        with self.assertRaises(StorageError):
            tx.query("users", where=[("age", ">", 0)], limit=-5)
        self.assertEqual(tx.commit(), self.engine.lsn)

    # ----------------------------------------------------- changed-back writes
    def test_changing_a_row_back_still_conflicts(self):
        tx = self.engine.begin(isolation="serializable")
        self.assertEqual(tx.get("users", 1)["age"], 30)
        self.engine.update("users", 1, {"age": 99})
        self.engine.update("users", 1, {"age": 30})  # back to the original value
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_other_transactions_uncommitted_or_rolled_back_are_ignored(self):
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        other = self.engine.begin()
        other.update("users", 1, {"age": 88})
        # uncommitted: no conflict yet, own snapshot unchanged
        self.assertEqual(tx.get("users", 1)["age"], 30)
        other.rollback()
        self.assertEqual(tx.commit(), self.engine.lsn)
        self.assertEqual(self.engine.get("users", 1)["age"], 30)

    def test_own_writes_do_not_trigger_a_read_conflict(self):
        tx = self.engine.begin(isolation="serializable")
        tx.query("users", where=[("age", ">=", 25)])
        tx.insert("users", {"id": 50, "name": "own", "age": 33})  # matches predicate
        tx.update("users", 1, {"age": 32})
        self.assertEqual(tx.commit(), self.engine.lsn)

    # ------------------------------------------------- cross serializable pair
    def test_two_serializable_transactions_cannot_both_commit(self):
        a = self.engine.begin(isolation="serializable")
        b = self.engine.begin(isolation="serializable")
        a.get("users", 1)
        b.get("users", 2)
        a.update("users", 2, {"age": 21})  # the row b read
        b.update("users", 1, {"age": 11})  # the row a read
        self.assertEqual(a.commit(), self.engine.lsn)
        with self.assertRaises(ReadConflictError):
            b.commit()
        self.assertEqual(b.state, "failed")
        self.assertEqual(self.engine.get("users", 1)["age"], 30)
        self.assertEqual(self.engine.get("users", 2)["age"], 21)  # a's write kept

    def test_conflict_independent_of_other_isolation_level(self):
        # a plain snapshot transaction still causes the serializable one to abort
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        snap = self.engine.begin()
        snap.update("users", 1, {"age": 60})
        snap.commit()
        with self.assertRaises(ReadConflictError):
            tx.commit()

    def test_conflict_independent_of_start_order(self):
        # the writer starts before the serializable reader but commits after it reads
        writer = self.engine.begin()
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        writer.update("users", 1, {"age": 70})
        writer.commit()
        with self.assertRaises(ReadConflictError):
            tx.commit()

    # ---------------------------------------------------------- abort semantics
    def test_conflict_fully_rolls_back_row_and_index_changes(self):
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        tx.insert("users", {"id": 70, "name": "tmp", "age": 70})
        tx.update("users", 2, {"age": 22})
        tx.delete("users", 3)
        self.engine.update("users", 1, {"age": 31})  # committed other; forces the conflict
        with self.assertRaises(ReadConflictError):
            tx.commit()
        self.assertEqual(tx.state, "failed")
        # tx's own changes are gone; the other transaction's commit survives
        self.assertEqual({r["id"]: r["age"] for r in self.engine.scan("users")},
                         {1: 31, 2: 20, 3: 40})
        self.assertIsNone(self.engine.get("users", 70))
        self.assertEqual(self.engine.get("users", 2)["age"], 20)
        self.assertEqual(self.engine.get("users", 3)["age"], 40)
        # index rebuilt after the aborted transaction retired
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 0, 99)], [2, 1, 3])
        self.assertTrue(self.engine.verify()["ok"])

    def test_no_audit_record_and_unchanged_lsn_on_conflict(self):
        # only a failed serializable tx, with no committing other transaction
        audit_count = len(self.engine.audit())
        lsn = self.engine.lsn
        other = self.engine.begin()
        other.update("users", 1, {"age": 31})
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        tx.update("users", 2, {"age": 22})
        other.commit()  # one new audit entry / lsn bump from the winner
        winner_audit = len(self.engine.audit())
        winner_lsn = self.engine.lsn
        with self.assertRaises(ReadConflictError):
            tx.commit()
        # the loser added no audit entry and moved no committed lsn
        self.assertEqual(len(self.engine.audit()), winner_audit)
        self.assertEqual(self.engine.lsn, winner_lsn)
        self.assertGreater(winner_audit, audit_count)
        self.assertGreater(winner_lsn, lsn)

    def test_terminated_transaction_rejects_further_use(self):
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ReadConflictError):
            tx.commit()
        for call in (
            lambda: tx.get("users", 1),
            lambda: tx.scan("users"),
            lambda: tx.insert("users", {"id": 71, "name": "x", "age": 1}),
            lambda: tx.commit(),
            lambda: tx.rollback(),
        ):
            with self.assertRaises(StorageError):
                call()

    def test_read_only_serializable_commits_without_conflict(self):
        tx = self.engine.begin(isolation="serializable")
        rows = tx.scan("users")
        lsn = tx.commit()
        self.assertEqual(lsn, self.engine.lsn)
        self.assertEqual(len(rows), 3)
        self.assertEqual(self.engine.audit()[-1]["ops"], [])

    # ------------------------------------------------------------ write-write
    def test_existing_write_write_conflict_still_raises_conflict_error(self):
        tx = self.engine.begin(isolation="serializable")
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ConflictError):
            tx.update("users", 1, {"age": 32})
        tx.rollback()

    # --------------------------------------------------------------- recovery
    def test_committed_serializable_state_survives_reopen(self):
        tx = self.engine.begin(isolation="serializable")
        tx.query("users", where=[("age", ">", 100)])  # empty predicate
        tx.insert("users", {"id": 80, "name": "persist", "age": 80})
        lsn = tx.commit()
        self.engine.reopen()
        self.assertEqual(self.engine.get("users", 80)["name"], "persist")
        self.assertEqual(self.engine.lsn, lsn)
        self.assertTrue(self.engine.verify()["ok"])

    def test_failed_serializable_transaction_is_invisible_after_reopen(self):
        tx = self.engine.begin(isolation="serializable")
        tx.get("users", 1)
        tx.insert("users", {"id": 81, "name": "ghost", "age": 81})
        self.engine.update("users", 1, {"age": 31})
        with self.assertRaises(ReadConflictError):
            tx.commit()
        self.engine.reopen()
        self.assertIsNone(self.engine.get("users", 81))
        self.assertEqual(self.engine.get("users", 1)["age"], 31)
        self.assertTrue(self.engine.verify()["ok"])

    def test_committed_serializable_state_survives_backup_restore(self):
        tx = self.engine.begin(isolation="serializable")
        tx.scan("users")
        tx.insert("users", {"id": 82, "name": "backup", "age": 82})
        tx.commit()
        backup_dir = os.path.join(self.root, "backup")
        self.engine.backup(backup_dir)
        restored = self.engine.restore(backup_dir)
        self.assertEqual(self.engine.get("users", 82)["name"], "backup")
        self.assertTrue(restored["restored_lsn"] > 0)


if __name__ == "__main__":
    unittest.main()
