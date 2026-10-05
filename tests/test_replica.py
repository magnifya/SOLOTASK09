"""Read-only replicas: creation, sync, pinned read sessions and refusals."""

import base64
import json
import os
import shutil
import tempfile
import unittest

from kvse import Engine, Replica, StorageError
from kvse.pager import DATA_NAME, WAL_NAME, read_wal
from kvse.replica import REPLICA_META, ReplicaSession


class ReplicaCase(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.mkdtemp(prefix="kvse-src-")
        self.dst = os.path.join(tempfile.mkdtemp(prefix="kvse-replica-parent-"), "replica")
        self.addCleanup(shutil.rmtree, self.src, True)
        self.addCleanup(shutil.rmtree, os.path.dirname(self.dst), True)
        self.engine = Engine(self.src, now_ms=1000)
        self.engine.create_table(
            "items",
            [
                {"name": "id", "type": "int", "nullable": False},
                {"name": "label", "type": "text"},
                {"name": "qty", "type": "int"},
            ],
            "id",
            indexes=["qty"],
        )
        self.lsn_table = self.engine.lsn

    def insert(self, pk, label, qty):
        self.engine.insert("items", {"id": pk, "label": label, "qty": qty})
        return self.engine.lsn

    def make_replica(self):
        return Replica.create(self.engine, self.dst)


class CreateTests(ReplicaCase):
    def test_create_returns_source_identity_and_applied_lsn(self):
        lsn = self.insert(1, "alpha", 5)
        replica = self.make_replica()
        self.assertIsInstance(replica, Replica)
        self.assertEqual(replica.source, self.engine.root)
        self.assertEqual(replica.applied_lsn, lsn)
        self.assertEqual(replica.info()["source"], self.engine.root)
        self.assertEqual(replica.info()["applied_lsn"], lsn)

    def test_create_accepts_a_plain_directory(self):
        self.insert(1, "alpha", 5)
        replica = Replica.create(self.src, self.dst)
        self.assertEqual(replica.source, os.path.abspath(self.src))
        self.assertEqual(replica.get("items", 1)["label"], "alpha")

    def test_create_on_empty_source(self):
        empty = tempfile.mkdtemp(prefix="kvse-empty-")
        self.addCleanup(shutil.rmtree, empty, True)
        Engine(empty)  # a real database directory with zero commits
        replica = Replica.create(empty, self.dst)
        self.assertEqual(replica.applied_lsn, 0)
        self.assertEqual(replica.list_tables(), [])

    def test_create_on_a_source_with_only_a_catalog(self):
        replica = self.make_replica()
        self.assertEqual(replica.applied_lsn, self.lsn_table)
        self.assertEqual(replica.list_tables(), ["items"])

    def test_replica_directory_is_self_contained(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        for name in (DATA_NAME, WAL_NAME, REPLICA_META):
            self.assertTrue(os.path.isfile(os.path.join(self.dst, name)), name)
        self.assertEqual(replica.scan("items"), self.engine.scan("items"))
        self.assertEqual(replica.audit(), self.engine.audit())
        self.assertTrue(replica.verify()["ok"])

    def test_replica_never_writes_back_to_the_source(self):
        self.insert(1, "alpha", 5)
        with open(os.path.join(self.src, WAL_NAME), "rb") as fh:
            wal_before = fh.read()
        with open(os.path.join(self.src, DATA_NAME), "rb") as fh:
            pages_before = fh.read()
        replica = self.make_replica()
        self.insert(2, "beta", 9)
        replica.sync()
        with open(os.path.join(self.src, WAL_NAME), "rb") as fh:
            wal_after = fh.read()
        with open(os.path.join(self.src, DATA_NAME), "rb") as fh:
            pages_after = fh.read()
        # The only source changes are the primary's own commit.
        self.assertTrue(wal_after.startswith(wal_before))
        self.assertNotEqual(wal_after, wal_before)
        self.assertEqual(len(pages_before), len(pages_after))
        # The replica holds its own independent copies.
        with open(os.path.join(self.dst, WAL_NAME), "rb") as fh:
            self.assertEqual(fh.read(), wal_after)
        self.assertNotEqual(os.path.abspath(self.dst), os.path.abspath(self.src))

    def test_create_rejects_bad_targets(self):
        with self.assertRaises(StorageError):
            Replica.create(self.engine, self.src)  # same directory as the source
        os.makedirs(self.dst)
        with open(os.path.join(self.dst, "junk"), "w") as fh:
            fh.write("x")
        with self.assertRaises(StorageError):
            Replica.create(self.engine, self.dst)  # not empty

    def test_create_rejects_a_missing_source(self):
        with self.assertRaises(StorageError):
            Replica.create(os.path.join(self.src, "nope"), self.dst)
        with self.assertRaises(StorageError):
            Replica.create(12345, self.dst)

    def test_open_requires_a_replica_directory(self):
        with self.assertRaises(StorageError):
            Replica(self.dst)  # never created
        with self.assertRaises(StorageError):
            Replica(self.src)  # a primary, not a replica


class ReadTests(ReplicaCase):
    def setUp(self):
        super().setUp()
        self.insert(1, "alpha", 5)
        self.insert(2, "beta", 9)
        self.insert(3, "gamma", 12)
        self.replica = self.make_replica()

    def test_reads_match_the_primary(self):
        engine, replica = self.engine, self.replica
        self.assertEqual(replica.get("items", 2), engine.get("items", 2))
        self.assertEqual(replica.scan("items"), engine.scan("items"))
        self.assertEqual(replica.index_get("items", "qty", 9), engine.index_get("items", "qty", 9))
        self.assertEqual(
            replica.index_range("items", "qty", 5, 12), engine.index_range("items", "qty", 5, 12)
        )
        result = replica.query("items", where=[("qty", ">=", 5)], order_by="id")
        expected = engine.query("items", where=[("qty", ">=", 5)], order_by="id")
        self.assertEqual(result, expected)
        self.assertEqual(result["access"], "index")  # query access marker preserved
        self.assertEqual(
            replica.explain("items", where=[("qty", ">=", 5)]),
            engine.explain("items", where=[("qty", ">=", 5)]),
        )
        self.assertEqual(replica.audit(), engine.audit())
        self.assertEqual(replica.verify()["ok"], engine.verify()["ok"])
        self.assertEqual(replica.table_info("items"), engine.table_info("items"))

    def test_read_errors_match(self):
        with self.assertRaises(StorageError):
            self.replica.get("nope", 1)
        with self.assertRaises(StorageError):
            self.replica.query("items", where=[("qty", "~~", 5)])

    def test_writes_are_refused_without_side_effects(self):
        replica = self.replica
        wal_records_before = replica.verify()["wal_records"]
        audit_before = replica.audit()
        for call in (
            lambda: replica.insert("items", {"id": 9, "label": "x", "qty": 1}),
            lambda: replica.update("items", 1, {"qty": 2}),
            lambda: replica.delete("items", 1),
            lambda: replica.create_table("t2", [{"name": "id", "type": "int"}], "id"),
            lambda: replica.create_index("items", "label"),
            lambda: replica.transaction([{"op": "delete", "table": "items", "pk": 1}]),
            lambda: replica.begin(),
            lambda: replica.restore(self.src),
            lambda: replica.checkpoint(),
            lambda: replica.commit(),
        ):
            with self.assertRaises(StorageError):
                call()
        self.assertEqual(replica.verify()["wal_records"], wal_records_before)
        self.assertEqual(replica.audit(), audit_before)
        self.assertEqual(replica.scan("items"), self.engine.scan("items"))


class SyncTests(ReplicaCase):
    def test_sync_without_target_tracks_the_latest_commit(self):
        first = self.insert(1, "alpha", 5)
        replica = self.make_replica()
        second = self.insert(2, "beta", 9)
        report = replica.sync()
        self.assertTrue(report["synced"])
        self.assertEqual(report["previous_lsn"], first)
        self.assertEqual(report["target_lsn"], second)
        self.assertEqual(report["applied_lsn"], second)
        self.assertEqual(replica.applied_lsn, second)
        self.assertEqual(replica.scan("items"), self.engine.scan("items"))

    def test_sync_without_new_commits_is_a_noop(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        report = replica.sync()
        self.assertFalse(report["synced"])
        self.assertEqual(report["applied_lsn"], replica.applied_lsn)

    def test_sync_to_an_explicit_boundary_ignores_newer_commits(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        second = self.insert(2, "beta", 9)
        self.insert(3, "gamma", 12)
        report = replica.sync(second)
        self.assertTrue(report["synced"])
        self.assertEqual(replica.applied_lsn, second)
        self.assertIsNone(replica.get("items", 3))  # the newer commit is not mixed in
        self.assertEqual([row["id"] for row in replica.scan("items")], [1, 2])
        # A later sync with no target catches up to the newest boundary.
        replica.sync()
        self.assertEqual(replica.get("items", 3)["label"], "gamma")

    def test_sync_rejects_bad_targets(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        latest = self.engine.lsn
        with self.assertRaises(StorageError):
            replica.sync(latest + 100)  # beyond the available history
        with self.assertRaises(StorageError):
            replica.sync(latest - 1)  # not a commit boundary
        with self.assertRaises(StorageError):
            replica.sync("not-a-number")
        self.assertEqual(replica.applied_lsn, latest)

    def test_sync_rejects_a_target_behind_the_replica(self):
        first = self.insert(1, "alpha", 5)
        self.insert(2, "beta", 9)
        replica = self.make_replica()
        with self.assertRaises(StorageError):
            replica.sync(first)
        self.assertEqual(replica.applied_lsn, self.engine.lsn)

    def test_sync_after_source_checkpoint(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        old_lsn = replica.applied_lsn
        self.engine.pager.checkpoint()  # the source drops its commit records
        with self.assertRaises(StorageError):
            replica.sync(old_lsn)  # no longer provided by the source
        self.assertEqual(replica.applied_lsn, old_lsn)
        # New commits are still syncable: every commit is a complete image.
        self.insert(2, "beta", 9)
        report = replica.sync()
        self.assertTrue(report["synced"])
        self.assertEqual(replica.scan("items"), self.engine.scan("items"))

    def test_failed_sync_leaves_the_replica_untouched(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        applied = replica.applied_lsn
        with open(os.path.join(self.dst, WAL_NAME), "rb") as fh:
            wal_before = fh.read()
        # Append a corrupt committed record to the source WAL.
        records, _ = read_wal(os.path.join(self.src, WAL_NAME))
        next_lsn = max(int(r["lsn"]) for r in records) + 1
        bad = {
            "lsn": next_lsn,
            "txid": 999,
            "page_id": 0,
            "payload_b64": base64.b64encode(b"\x00" * 4088).decode("ascii"),
            "crc32": 1,  # wrong on purpose
        }
        marker = {"lsn": next_lsn + 1, "txid": 999, "commit": True}
        with open(os.path.join(self.src, WAL_NAME), "ab") as fh:
            for record in (bad, marker):
                fh.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("items", 1)["label"], "alpha")
        with open(os.path.join(self.dst, WAL_NAME), "rb") as fh:
            self.assertEqual(fh.read(), wal_before)

    def test_sync_rejects_corrupt_source_pages(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        applied = replica.applied_lsn
        self.insert(2, "beta", 9)
        data_path = os.path.join(self.src, DATA_NAME)
        with open(data_path, "r+b") as fh:
            blob = bytearray(fh.read())
            blob[-1] ^= 0xFF  # break the last page's checksum
            fh.seek(0)
            fh.write(bytes(blob))
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, applied)
        self.assertIsNone(replica.get("items", 2))

    def test_restart_keeps_applied_lsn_and_catches_up(self):
        self.insert(1, "alpha", 5)
        replica = self.make_replica()
        applied = replica.applied_lsn
        self.insert(2, "beta", 9)
        reopened = Replica(self.dst)  # a fresh process would do exactly this
        self.assertEqual(reopened.applied_lsn, applied)
        self.assertEqual(reopened.source, self.engine.root)
        report = reopened.sync()
        self.assertTrue(report["synced"])
        self.assertEqual(reopened.scan("items"), self.engine.scan("items"))
        again = Replica(self.dst)
        self.assertEqual(again.applied_lsn, self.engine.lsn)


class SessionTests(ReplicaCase):
    def setUp(self):
        super().setUp()
        self.lsn_one = self.insert(1, "alpha", 5)
        self.lsn_two = self.insert(2, "beta", 9)
        self.replica = self.make_replica()

    def test_session_pins_its_boundary_across_syncs(self):
        session = self.replica.read_session()
        self.assertIsInstance(session, ReplicaSession)
        self.assertEqual(session.applied_lsn, self.lsn_two)
        self.insert(3, "gamma", 12)
        self.replica.sync()
        # The session still sees its creation-time boundary.
        self.assertIsNone(session.get("items", 3))
        self.assertEqual([row["id"] for row in session.scan("items")], [1, 2])
        self.assertEqual(self.replica.get("items", 3)["label"], "gamma")
        session.close()

    def test_session_at_an_explicit_lsn(self):
        session = self.replica.read_session(self.lsn_one)
        self.assertEqual([row["id"] for row in session.scan("items")], [1])
        self.assertIsNone(session.get("items", 2))
        # Queries, predicates and access markers match the primary at that lsn.
        result = session.query("items", where=[("qty", ">=", 1)])
        self.assertEqual(result["access"], "index")
        self.assertEqual([row["id"] for row in result["rows"]], [1])
        self.assertEqual(session.explain("items", where=[("qty", ">=", 1)])["access"], "index")
        self.assertEqual(len(session.audit()), 2)  # create_table + first insert
        self.assertTrue(session.verify()["ok"])
        session.close()

    def test_session_survives_replica_reopen(self):
        session = self.replica.read_session(self.lsn_one)
        self.insert(3, "gamma", 12)
        self.replica.sync()
        Replica(self.dst)  # simulate a replica restart
        self.assertEqual([row["id"] for row in session.scan("items")], [1])
        session.close()

    def test_session_beyond_the_applied_boundary_raises(self):
        with self.assertRaises(StorageError):
            self.replica.read_session(self.replica.applied_lsn + 1)
        with self.assertRaises(StorageError):
            self.replica.read_session(-1)
        with self.assertRaises(StorageError):
            self.replica.read_session(self.lsn_two - 1)  # not a commit boundary

    def test_session_is_read_only_and_closeable(self):
        session = self.replica.read_session()
        with self.assertRaises(StorageError):
            session.insert("items", {"id": 9, "label": "x", "qty": 1})
        with self.assertRaises(StorageError):
            session.transaction([{"op": "delete", "table": "items", "pk": 1}])
        session.close()
        with self.assertRaises(StorageError):
            session.get("items", 1)
        with self.replica.read_session() as scoped:
            self.assertEqual(scoped.get("items", 1)["label"], "alpha")


if __name__ == "__main__":
    unittest.main()
