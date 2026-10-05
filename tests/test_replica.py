"""Replica tests: initial copy, bounded sync, read sessions and refusal."""

import json
import os
import shutil
import tempfile
import threading
import unittest

from kvse import Engine, Replica, StorageError

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
]


class ReplicaTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="kvse-replica-")
        self.primary_dir = os.path.join(self.base, "primary")
        self.replica_dir = os.path.join(self.base, "replica")
        self.primary = Engine(self.primary_dir, now_ms=lambda: 1000)
        self.primary.create_table("users", COLUMNS, "id", indexes=["age", "name"])

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def add(self, pk, name, age):
        return self.primary.insert("users", {"id": pk, "name": name, "age": age})

    # ------------------------------------------------------------ creation
    def test_initial_replica_reports_source_and_lsn(self):
        self.add(1, "ann", 30)
        self.add(2, "bob", 20)
        lsn = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.assertEqual(replica.source_id, os.path.realpath(self.primary_dir))
        self.assertEqual(replica.applied_lsn, lsn)
        self.assertEqual(replica.info()["applied_lsn"], lsn)
        self.assertEqual(replica.lsn, lsn)
        self.assertEqual([r["id"] for r in replica.scan("users")], [1, 2])

        # independent storage: its own directory, no shared files
        self.assertTrue(os.path.isfile(os.path.join(self.replica_dir, "replica.json")))
        gen = [n for n in os.listdir(self.replica_dir) if n.startswith("gen-")]
        self.assertEqual(len(gen), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.replica_dir, gen[0], "data.pages")))
        self.assertTrue(os.path.isfile(os.path.join(self.replica_dir, gen[0], "wal.log")))

        # creating into an existing replica dir is refused rather than rebuilt
        with self.assertRaises(StorageError):
            Replica.create(self.primary, self.replica_dir)

    def test_create_from_directory_source_and_open(self):
        self.add(1, "ann", 30)
        lsn = self.primary.lsn
        replica = Replica.create(self.primary_dir, self.replica_dir)
        self.assertEqual(replica.applied_lsn, lsn)
        again = Replica.open(self.replica_dir)
        self.assertEqual(again.applied_lsn, lsn)
        self.assertEqual(again.get("users", 1)["name"], "ann")
        with self.assertRaises(StorageError):
            Replica.open(os.path.join(self.base, "missing"))

    def test_primary_is_not_modified_by_replica_operations(self):
        self.add(1, "ann", 30)
        lsn = self.primary.lsn
        primary_files = set(os.listdir(self.primary_dir))
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        replica.sync()
        self.assertEqual(set(os.listdir(self.primary_dir)), primary_files)
        self.assertGreater(self.primary.lsn, lsn)
        # replica files never appear in the primary directory
        self.assertFalse(os.path.exists(os.path.join(self.primary_dir, "replica.json")))

    def test_invalid_sources_and_targets(self):
        with self.assertRaises(StorageError):
            Replica.create(123, self.replica_dir)
        with self.assertRaises(StorageError):
            Replica.create(self.primary, self.primary_dir)  # same directory
        with self.assertRaises(StorageError):
            Replica.create(os.path.join(self.base, "no-such-db"), self.replica_dir)

    # --------------------------------------------------------------- sync
    def test_sync_follows_commit_boundaries_in_order(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.assertEqual(replica.applied_lsn, lsn1)

        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        self.primary.update("users", 1, {"age": 31})
        lsn3 = self.primary.lsn

        report = replica.sync(target_lsn=lsn2)
        self.assertTrue(report["applied"])
        self.assertEqual((report["from_lsn"], report["applied_lsn"]), (lsn1, lsn2))
        self.assertEqual([r["id"] for r in replica.scan("users")], [1, 2])
        self.assertEqual(replica.get("users", 1)["age"], 30)

        report = replica.sync()
        self.assertEqual(report["applied_lsn"], lsn3)
        self.assertTrue(report["caught_up"])
        self.assertEqual(replica.get("users", 1)["age"], 31)

        # syncing again is a no-op
        report = replica.sync()
        self.assertFalse(report["applied"])
        self.assertEqual(report["applied_lsn"], lsn3)

    def test_explicit_sync_validates_the_target(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn

        with self.assertRaises(StorageError):
            replica.sync(target_lsn=lsn1 - 1)          # older than applied
        with self.assertRaises(StorageError):
            replica.sync(target_lsn=lsn2 + 100)        # beyond history
        with self.assertRaises(StorageError):
            replica.sync(target_lsn=lsn1 + 1)          # not a commit boundary
        for bad in ("2", 1.5, True, [1], object()):
            with self.assertRaises(StorageError):
                replica.sync(target_lsn=bad)
        self.assertEqual(replica.applied_lsn, lsn1)

    def test_sync_never_mixes_in_concurrent_commits(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        self.add(3, "cid", 40)
        lsn3 = self.primary.lsn
        # pin to lsn2 explicitly even though lsn3 already exists
        report = replica.sync(target_lsn=lsn2)
        self.assertEqual(report["applied_lsn"], lsn2)
        self.assertEqual([r["id"] for r in replica.scan("users")], [1, 2])
        self.assertIsNone(replica.get("users", 3))
        self.assertEqual(replica.source_lsn(), lsn3)

    def test_failed_sync_leaves_previous_state_intact(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        rows_before = replica.scan("users")
        self.add(2, "bob", 20)

        with self.assertRaises(StorageError):
            replica.sync(target_lsn=self.primary.lsn + 50)
        with self.assertRaises(StorageError):
            replica.sync(target_lsn=lsn1 + 1)
        self.assertEqual(replica.applied_lsn, lsn1)
        self.assertEqual(replica.scan("users"), rows_before)
        self.assertTrue(replica.verify()["ok"])
        # no staging directories are left behind
        self.assertFalse(
            any(n.startswith("stage-") for n in os.listdir(self.replica_dir))
        )
        with open(os.path.join(self.replica_dir, "replica.json"), "rb") as fh:
            meta = json.loads(fh.read())
        self.assertEqual(meta["applied_lsn"], lsn1)

        # the replica still works for a good sync afterwards
        good = replica.sync()
        self.assertEqual(good["applied_lsn"], self.primary.lsn)
        self.assertEqual([r["id"] for r in replica.scan("users")], [1, 2])

    def test_applied_lsn_survives_restart_and_continues_catching_up(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        replica.sync(target_lsn=lsn2)
        report = replica.reopen()
        self.assertEqual(report["applied_lsn"], lsn2)
        reloaded = Replica.open(self.replica_dir)
        self.assertEqual(reloaded.applied_lsn, lsn2)
        self.assertEqual([r["id"] for r in reloaded.scan("users")], [1, 2])
        self.assertTrue(reloaded.verify()["ok"])

        self.primary.update("users", 1, {"age": 99})
        lsn3 = self.primary.lsn
        report = reloaded.sync()
        self.assertEqual((report["from_lsn"], report["applied_lsn"]), (lsn2, lsn3))
        self.assertEqual(reloaded.get("users", 1)["age"], 99)

    def test_replica_rejects_a_different_source(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        other_dir = os.path.join(self.base, "other")
        other = Engine(other_dir)
        other.create_table("users", COLUMNS, "id")
        with self.assertRaises(StorageError):
            Replica(other, self.replica_dir)  # wrong source identity
        with self.assertRaises(StorageError):
            Replica(other_dir + "-nope", self.replica_dir)
        replica.sync(target_lsn=replica.applied_lsn)  # original source still fine

    # ----------------------------------------------------------- read parity
    def test_reads_queries_indexes_explain_match_the_primary(self):
        for pk, name, age in [(1, "ann", 30), (2, "bob", 20), (3, "cid", 40)]:
            self.add(pk, name, age)
        replica = self.primary.create_replica(self.replica_dir)
        boundary_scan = replica.scan("users")
        boundary_get2 = replica.get("users", 2)
        self.primary.delete("users", 2)
        # the replica keeps serving the primary state at its boundary
        self.assertEqual(replica.scan("users"), boundary_scan)
        self.assertEqual(replica.get("users", 2), boundary_get2)

        # rewind the primary to the same boundary: every read path must agree
        self.primary.insert("users", boundary_get2)
        self.assertEqual(replica.get("users", 2), self.primary.get("users", 2))
        self.assertEqual(
            sorted(r["id"] for r in replica.scan("users")),
            sorted(r["id"] for r in self.primary.scan("users")),
        )
        self.assertEqual(
            [r["id"] for r in replica.index_get("users", "name", "ann")],
            [r["id"] for r in self.primary.index_get("users", "name", "ann")],
        )
        self.assertEqual(
            [r["id"] for r in replica.index_range("users", "age", 25, 45)],
            [r["id"] for r in self.primary.index_range("users", "age", 25, 45)],
        )
        self.assertEqual(
            replica.query("users", where=[("age", ">=", 25)]),
            self.primary.query("users", where=[("age", ">=", 25)]),
        )
        self.assertEqual(
            replica.explain("users", where=[("age", ">=", 1)]),
            self.primary.explain("users", where=[("age", ">=", 1)]),
        )
        result = replica.query(
            "users", columns=["id", "age"], where=[("name", "!=", "ann")], order_by="id"
        )
        self.assertEqual(result["access"], "scan")
        self.assertEqual(result["rows"], [{"id": 2, "age": 20}, {"id": 3, "age": 40}])
        # predicate/access errors on the replica match the primary's
        with self.assertRaises(StorageError):
            replica.query("users", where=[("nope", "=", 1)])
        with self.assertRaises(StorageError):
            replica.index_range("users", "nope", 0, 1)

    def test_audit_is_copied_and_independent(self):
        self.add(1, "ann", 30)
        self.add(2, "bob", 20)
        replica = self.primary.create_replica(self.replica_dir)
        entries = replica.audit()
        self.assertEqual(len(entries), 3)  # create_table + two inserts
        self.assertEqual(entries[-1]["ops"][0]["op"], "insert")
        self.add(3, "cid", 40)
        self.assertEqual(len(replica.audit()), 3)
        replica.sync()
        self.assertEqual(len(replica.audit()), 4)
        self.assertEqual(replica.audit(limit=1)[0]["ops"][0]["row"]["id"], 3)

    # ---------------------------------------------------------- read sessions
    def test_read_session_pins_a_boundary_across_syncs(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        replica.sync()
        self.primary.update("users", 1, {"age": 31})
        lsn3 = self.primary.lsn
        replica.sync()

        session = replica.read_session(lsn1)
        self.assertEqual(session.lsn, lsn1)
        self.assertEqual([r["id"] for r in session.scan("users")], [1])
        self.assertEqual(session.get("users", 1)["age"], 30)

        session2 = replica.read_session(lsn2)
        self.assertEqual(sorted(r["id"] for r in session2.scan("users")), [1, 2])
        self.assertEqual(session2.get("users", 1)["age"], 30)

        latest = replica.read_session()
        self.assertEqual(latest.lsn, lsn3)
        self.assertEqual(latest.get("users", 1)["age"], 31)

        with self.assertRaises(StorageError):
            replica.read_session(lsn3 + 1)
        with self.assertRaises(StorageError):
            replica.read_session(lsn1 + 1)  # not a commit boundary
        with self.assertRaises(StorageError):
            replica.read_session(-1)

    def test_session_survives_a_later_sync_and_context_close(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        replica.sync()
        with replica.read_session() as session:
            pinned = session.lsn
            self.primary.update("users", 1, {"age": 50})
            lsn3 = self.primary.lsn
            replica.sync()
            self.assertEqual(session.lsn, pinned)
            self.assertEqual(session.get("users", 1)["age"], 30)
            self.assertEqual([r["id"] for r in session.scan("users")], [1, 2])
        self.assertEqual(lsn3, replica.applied_lsn)
        with self.assertRaises(StorageError):
            session.get("users", 1)

    def test_session_supports_the_full_read_surface(self):
        self.add(1, "ann", 30)
        self.add(2, "bob", 20)
        replica = self.primary.create_replica(self.replica_dir)
        session = replica.read_session()
        self.assertEqual(session.table_info("users")["rows"], 2)
        self.assertTrue(session.has_table("users"))
        self.assertEqual(session.list_tables(), ["users"])
        self.assertEqual(
            [r["id"] for r in session.index_range("users", "age", None, 25)], [2]
        )
        self.assertTrue(session.verify()["ok"])
        self.assertEqual(len(session.audit()), 3)
        self.assertEqual(
            session.explain("users", where=[("age", ">", 1)])["access"], "index"
        )

    def test_session_keeps_its_generation_until_closed(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        old_gen = [n for n in os.listdir(self.replica_dir) if n.startswith("gen-")][0]
        session = replica.read_session()  # pins the initial boundary
        self.add(2, "bob", 20)
        replica.sync()
        new_gen = replica._gen
        self.assertNotEqual(old_gen, new_gen)
        # the pinned generation stays on disk while the session is open
        self.assertTrue(os.path.isdir(os.path.join(self.replica_dir, old_gen)))
        self.assertEqual(session.get("users", 1)["name"], "ann")
        self.assertIsNone(session.get("users", 2))
        session.close()
        self.assertFalse(os.path.exists(os.path.join(self.replica_dir, old_gen)))
        with self.assertRaises(StorageError):
            session.get("users", 1)

    def test_concurrent_syncs_and_reads_stay_consistent(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    rows = replica.scan("users")
                    self.assertTrue(all("name" in row for row in rows))
                    replica.verify()
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for pk in range(2, 12):
            self.add(pk, "user-%d" % pk, 20 + pk)
            replica.sync()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(len(replica.scan("users")), 11)
        self.assertTrue(replica.verify()["ok"])

    def test_session_at_lsn_zero_only_for_an_empty_replica(self):
        empty_primary = Engine(os.path.join(self.base, "empty-p"))
        empty_replica = Replica.create(
            empty_primary, os.path.join(self.base, "empty-r")
        )
        self.assertEqual(empty_replica.applied_lsn, 0)
        session = empty_replica.read_session(0)
        self.assertEqual(session.list_tables(), [])
        session.close()
        # once the primary has commits the pre-history (lsn 0) cannot be
        # rebuilt from WAL, exactly like point-in-time restore
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        with self.assertRaises(StorageError):
            replica.read_session(0)

    # ------------------------------------------------------------- refusal
    def test_writes_are_refused_without_wal_or_audit(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        audit_before = len(replica.audit())
        wal_size = os.path.getsize(
            os.path.join(self.replica_dir,
                         [n for n in os.listdir(self.replica_dir) if n.startswith("gen-")][0],
                         "wal.log")
        )
        refused = [
            lambda: replica.insert("users", {"id": 9, "name": "z", "age": 1}),
            lambda: replica.update("users", 1, {"age": 1}),
            lambda: replica.delete("users", 1),
            lambda: replica.create_table("t", COLUMNS, "id"),
            lambda: replica.create_index("users", "id"),
            lambda: replica.begin(),
            lambda: replica.commit(),
            lambda: replica.rollback(),
            lambda: replica.transaction([{"op": "delete", "table": "users", "pk": 1}]),
            lambda: replica.restore("."),
        ]
        for call in refused:
            with self.assertRaises(StorageError):
                call()
        self.assertEqual(len(replica.audit()), audit_before)
        gen = [n for n in os.listdir(self.replica_dir) if n.startswith("gen-")][0]
        self.assertEqual(
            os.path.getsize(os.path.join(self.replica_dir, gen, "wal.log")), wal_size
        )
        self.assertEqual(replica.get("users", 1)["name"], "ann")

        session = replica.read_session()
        for call in [
            lambda: session.insert("users", {"id": 9, "name": "z", "age": 1}),
            lambda: session.update("users", 1, {"age": 1}),
            lambda: session.delete("users", 1),
            lambda: session.create_table("t", COLUMNS, "id"),
            lambda: session.begin(),
            lambda: session.restore("."),
        ]:
            with self.assertRaises(StorageError):
                call()

    # ------------------------------------------------------ corruption/checkpoint
    def test_checkpointed_primary_is_a_valid_source(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        self.primary.pager.checkpoint()  # WAL truncated; in-memory lsn keeps rising
        replica = self.primary.create_replica(self.replica_dir)
        self.assertEqual(replica.applied_lsn, lsn1)
        self.assertEqual(replica.get("users", 1)["name"], "ann")
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        report = replica.sync()
        self.assertEqual(report["applied_lsn"], lsn2)
        self.assertEqual([r["id"] for r in replica.scan("users")], [1, 2])

        # a fresh replica can also be seeded straight from the image boundary
        fresh = Replica.create(self.primary, os.path.join(self.base, "replica2"))
        self.assertEqual(fresh.applied_lsn, lsn2)

    def test_checkpoint_then_primary_restart_rewinds_history(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        applied = replica.applied_lsn
        self.primary.pager.checkpoint()
        self.primary.reopen()  # the lsn sequence restarts on the primary
        self.add(2, "bob", 20)
        # the source history can no longer be ordered against the replica
        with self.assertRaises(StorageError):
            replica.sync()
        # failure is atomic: the replica keeps its old boundary
        self.assertEqual(replica.applied_lsn, applied)
        self.assertEqual(replica.get("users", 1)["name"], "ann")
        # a brand new replica can seed from the current image instead
        fresh = Replica.create(self.primary, os.path.join(self.base, "replica2"))
        self.assertEqual([r["id"] for r in fresh.scan("users")], [1, 2])

    def test_old_layout_directory_needs_no_migration(self):
        self.add(1, "ann", 30)
        # a pristine primary dir contains only the original two files
        self.assertEqual(
            sorted(os.listdir(self.primary_dir)), ["data.pages", "wal.log"]
        )
        replica = Replica(self.primary_dir, self.replica_dir)
        self.assertEqual(replica.get("users", 1)["name"], "ann")

    def test_corrupt_source_page_is_rejected_atomically(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        # corrupt a source payload page (page 2, after header)
        with open(os.path.join(self.primary_dir, "data.pages"), "r+b") as fh:
            fh.seek(4096 * 2 + 100)
            fh.write(b"\xff\xff\xff")
        with self.assertRaises(StorageError):
            replica.sync()
        self.assertEqual(replica.applied_lsn, lsn1)
        self.assertTrue(replica.verify()["ok"])
        self.assertFalse(
            any(n.startswith("stage-") for n in os.listdir(self.replica_dir))
        )

    def test_corrupt_source_wal_record_is_rejected(self):
        self.add(1, "ann", 30)
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        records_path = os.path.join(self.primary_dir, "wal.log")
        with open(records_path, "rb") as fh:
            blob = fh.read()
        lines = blob.splitlines(keepends=True)
        # corrupt the checksum of the last page-write record (part of commit 2)
        idx = max(i for i, line in enumerate(lines) if b'"page_id"' in line)
        record = json.loads(lines[idx])
        record["crc32"] = int(record["crc32"]) ^ 0xFFFFFFFF
        lines[idx] = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with open(records_path, "wb") as fh:
            fh.write(b"".join(lines))
        with self.assertRaises(StorageError):
            replica.sync(target_lsn=lsn2)
        self.assertEqual(replica.scan("users"), [{"id": 1, "name": "ann", "age": 30}])
        self.assertEqual(replica.applied_lsn, replica.info()["applied_lsn"])

    def test_sync_after_source_checkpoint_rejects_out_of_reach_target(self):
        self.add(1, "ann", 30)
        lsn1 = self.primary.lsn
        replica = self.primary.create_replica(self.replica_dir)
        self.add(2, "bob", 20)
        lsn2 = self.primary.lsn
        self.add(3, "cid", 40)
        lsn3 = self.primary.lsn
        self.primary.pager.checkpoint()  # truncates WAL; image now holds lsn3
        # can catch up to the image boundary ...
        report = replica.sync(target_lsn=lsn3)
        self.assertEqual(report["applied_lsn"], lsn3)
        # ... but the intermediate boundary is no longer in the source WAL
        fresh = self.primary.create_replica(
            os.path.join(self.base, "replica2")
        )
        self.assertEqual(fresh.applied_lsn, lsn3)
        with self.assertRaises(StorageError):
            fresh.sync(target_lsn=lsn2)
        with self.assertRaises(StorageError):
            fresh.sync(target_lsn=lsn1)


if __name__ == "__main__":
    unittest.main()
