"""Tests for the paged B+ tree indexes: structure, persistence, recovery."""

import json
import os
import shutil
import struct
import tempfile
import unittest

from kvse import Engine, Replica, StorageError
from kvse.btree import BPlusTree, build_pages, entry_key, read_entries
from kvse.pager import PAGE_SIZE, Pager, crc32

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
    {"name": "active", "type": "bool"},
]


def make_engine(root):
    engine = Engine(root, now_ms=lambda: 1000)
    engine.create_table("users", COLUMNS, "id", indexes=["name", "age"])
    return engine


def fill(engine, count):
    tx = engine.begin()
    for i in range(count):
        tx.insert("users", {"id": i, "name": "user-%04d" % i, "age": (i * 37) % 100})
    return tx.commit()


def read_state(engine):
    """Decode the persisted state blob of ``engine``'s directory."""
    meta = json.loads(engine.pager.read_page(0).rstrip(b"\x00").decode("utf-8"))
    blob = b"".join(
        engine.pager.read_page(meta["state_page"] + i) for i in range(meta["state_pages"])
    )
    return meta, json.loads(blob[: meta["state_len"]].decode("utf-8"))


def index_refs(engine):
    return read_state(engine)[1]["index_refs"]


def rewrite_state(engine, mutate):
    """Replace the persisted state blob directly in the page file."""
    engine.pager.checkpoint()  # empty the WAL so the mutated image is what loads
    meta, state = read_state(engine)
    mutate(state)
    blob = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_size = engine.pager.payload_size
    meta["state_pages"] = max(1, -(-len(blob) // payload_size))
    meta["state_len"] = len(blob)
    pages = {0: json.dumps(meta, sort_keys=True).encode("utf-8")}
    for i in range(meta["state_pages"]):
        pages[meta["state_page"] + i] = blob[i * payload_size:(i + 1) * payload_size]
    with open(engine.pager.data_path, "r+b") as handle:
        for page_id, payload in pages.items():
            block = payload.ljust(payload_size, b"\x00")
            handle.seek(page_id * engine.page_size)
            handle.write(struct.pack(">II", page_id, crc32(block)) + block)
        handle.flush()
        os.fsync(handle.fileno())


def corrupt_page(engine, page_id):
    with open(engine.pager.data_path, "r+b") as handle:
        handle.seek(page_id * engine.page_size + 100)
        handle.write(b"X")


class InMemoryTreeTests(unittest.TestCase):
    def test_insert_keeps_key_order_and_splits(self):
        tree = BPlusTree(order=8)
        entries = [((i * 53) % 97, i) for i in range(500)]
        for entry in entries:
            tree.insert(entry)
        self.assertEqual(len(tree), 500)
        self.assertEqual(list(tree), sorted(entries, key=entry_key))
        self.assertGreater(len(list(self._leaves(tree))), 1)  # pages split

    def _leaves(self, tree):
        node = tree.root
        while not hasattr(node, "entries"):
            node = node.children[0]
        while node is not None:
            yield node
            node = node.next

    def test_duplicates_are_ignored(self):
        tree = BPlusTree(order=4)
        for _ in range(3):
            tree.insert((7, 1))
        self.assertEqual(list(tree), [(7, 1)])
        self.assertEqual(len(tree), 1)

    def test_range_bounds_are_closed_and_none_is_open(self):
        tree = BPlusTree(order=4)
        for i in range(100):
            tree.insert((i, i))
        self.assertEqual([v for v, _ in tree.iter_range(10, 15)], list(range(10, 16)))
        self.assertEqual([v for v, _ in tree.iter_range(None, 2)], [0, 1, 2])
        self.assertEqual([v for v, _ in tree.iter_range(97, None)], [97, 98, 99])
        self.assertEqual(list(tree.iter_range(50, 40)), [])  # empty range
        self.assertEqual(list(tree.iter_range(1000, 2000)), [])

    def test_bulk_load_matches_incremental_order(self):
        tree = BPlusTree(order=5)
        entries = [("k%03d" % ((i * 29) % 101), i) for i in range(300)]
        tree.bulk_load(entries)
        self.assertEqual(list(tree), sorted(entries, key=entry_key))
        want = sorted(v for v, _ in entries if "k010" <= v <= "k020")
        self.assertEqual([v for v, _ in tree.iter_range("k010", "k020")], want)


class PagedTreeTests(unittest.TestCase):
    def test_page_round_trip_multi_level(self):
        entries = [(i * 7 % 500, i) for i in range(3000)]
        entries.sort(key=entry_key)
        root = tempfile.mkdtemp(prefix="kvse-btree-")
        try:
            pager = Pager(root)
            payloads, root_page, height = build_pages(
                "t", "c", entries, 1, pager.payload_size
            )
            self.assertGreater(height, 1)  # the index spans internal pages
            for offset, payload in enumerate(payloads):
                pager.write_page(1, 1 + offset, payload)
            pager.commit(1)
            loaded = read_entries(pager.read_page, root_page, "t", "c", pager.page_count())
            self.assertEqual([tuple(e) for e in loaded], entries)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_read_entries_rejects_bad_structures(self):
        root = tempfile.mkdtemp(prefix="kvse-btree-")
        try:
            pager = Pager(root)
            entries = [(i, i) for i in range(50)]
            payloads, root_page, _ = build_pages("t", "c", entries, 1, pager.payload_size)
            for offset, payload in enumerate(payloads):
                pager.write_page(1, 1 + offset, payload)
            pager.commit(1)
            with self.assertRaises(StorageError):
                read_entries(pager.read_page, 999, "t", "c", pager.page_count())
            with self.assertRaises(StorageError):
                read_entries(pager.read_page, root_page, "other", "c", pager.page_count())
        finally:
            shutil.rmtree(root, ignore_errors=True)


class EngineIndexTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-btree-engine-")
        self.engine = make_engine(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_multi_page_index_survives_reopen(self):
        fill(self.engine, 800)
        refs = index_refs(self.engine)
        self.assertEqual({(r["table"], r["column"]) for r in refs},
                         {("users", "age"), ("users", "name")})
        pages_before = self.engine.pager.page_count()
        self.assertGreater(pages_before, 4)  # index pages beyond meta+audit+state

        expected_range = [r["id"] for r in self.engine.index_range("users", "age", 10, 20)]
        expected_get = self.engine.index_get("users", "name", "user-0400")
        self.engine.reopen()
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", 10, 20)],
                         expected_range)
        self.assertEqual(self.engine.index_get("users", "name", "user-0400"), expected_get)
        self.assertTrue(self.engine.verify()["ok"])

    def test_index_order_is_key_then_pk(self):
        tx = self.engine.begin()
        for pk, age in [(5, 30), (1, 30), (3, 10), (2, 30), (4, 10)]:
            tx.insert("users", {"id": pk, "name": "n%d" % pk, "age": age})
        tx.commit()
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age", None, None)],
                         [3, 4, 1, 2, 5])
        self.assertEqual([r["id"] for r in self.engine.index_get("users", "age", 30)], [1, 2, 5])

    def test_null_values_are_not_indexed(self):
        self.engine.insert("users", {"id": 1, "name": "a"})
        self.engine.insert("users", {"id": 2, "name": "b", "age": 5})
        self.assertEqual([r["id"] for r in self.engine.index_range("users", "age")], [2])
        refs = {r["column"]: r for r in index_refs(self.engine)}
        self.assertEqual(refs["age"]["entries"], 1)

    def test_updates_deletes_and_rollback_stay_consistent(self):
        fill(self.engine, 300)
        self.engine.update("users", 10, {"age": 1})
        self.engine.delete("users", 20)
        tx = self.engine.begin()
        tx.insert("users", {"id": 9999, "name": "ghost", "age": 1})
        tx.rollback()
        self.assertEqual(self.engine.index_get("users", "name", "ghost"), [])
        self.assertNotIn(20, [r["id"] for r in self.engine.index_range("users", "age")])
        self.engine.reopen()
        self.assertIn(10, [r["id"] for r in self.engine.index_get("users", "age", 1)])
        self.assertNotIn(9999, [r["id"] for r in self.engine.index_range("users", "age")])
        self.assertNotIn(20, [r["id"] for r in self.engine.index_range("users", "age")])
        self.assertTrue(self.engine.verify()["ok"])

    def test_crash_recovery_restores_index_pages(self):
        fill(self.engine, 400)
        expected = self.engine.index_range("users", "age", 0, 50)
        with open(self.engine.pager.data_path, "wb"):
            pass  # lose every page; only the WAL survives
        self.engine.reopen()
        self.assertEqual(self.engine.index_range("users", "age", 0, 50), expected)
        self.assertTrue(self.engine.verify()["ok"])

    def test_checkpoint_keeps_index_readable(self):
        fill(self.engine, 200)
        self.engine.pager.checkpoint()
        self.assertEqual(self.engine.pager.wal_records(), 0)
        self.engine.reopen()
        self.assertEqual(len(self.engine.index_range("users", "age", 0, 100)), 200)
        self.assertTrue(self.engine.verify()["ok"])

    def test_old_image_without_index_pages_still_opens(self):
        fill(self.engine, 120)
        before = self.engine.index_range("users", "age", 0, 100)
        rewrite_state(self.engine, lambda state: state.pop("index_refs"))
        reopened = Engine(self.root, now_ms=lambda: 1000)
        self.assertEqual(reopened.index_range("users", "age", 0, 100), before)
        self.assertEqual(reopened.index_get("users", "name", "user-0007"),
                         self.engine.index_get("users", "name", "user-0007"))
        self.assertTrue(reopened.verify()["ok"])
        # the next commit upgrades the image to paged indexes
        reopened.insert("users", {"id": 1000, "name": "new", "age": 1})
        self.assertEqual({(r["table"], r["column"]) for r in index_refs(reopened)},
                         {("users", "age"), ("users", "name")})
        reopened.reopen()
        self.assertTrue(reopened.verify()["ok"])

    def test_corrupt_index_page_fails_verify_and_reopen(self):
        fill(self.engine, 300)
        refs = index_refs(self.engine)
        root_page = refs[0]["root"]
        self.engine.pager.checkpoint()  # empty the WAL so replay cannot heal
        corrupt_page(self.engine, root_page)
        report = self.engine.verify()
        self.assertFalse(report["crc_ok"])
        self.assertFalse(report["ok"])
        self.assertEqual(sorted(report), ["crc_ok", "ok", "pages", "wal_records"])
        with self.assertRaises(StorageError):
            Engine(self.root)

    def test_broken_index_structure_fails_verify_and_reopen(self):
        fill(self.engine, 300)
        # point the age index root at the name index root: parses fine, but
        # ownership/order checks must reject it without a checksum error
        refs = index_refs(self.engine)
        roots = {ref["column"]: ref["root"] for ref in refs}

        def mutate(state):
            for ref in state["index_refs"]:
                if ref["column"] == "age":
                    ref["root"] = roots["name"]

        rewrite_state(self.engine, mutate)
        report = self.engine.verify()
        self.assertTrue(report["crc_ok"])
        self.assertFalse(report["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root)

    def test_out_of_range_index_pointer_is_rejected(self):
        fill(self.engine, 50)

        def mutate(state):
            state["index_refs"][0]["root"] = 10 ** 6

        rewrite_state(self.engine, mutate)
        self.assertFalse(self.engine.verify()["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root)

    def test_backup_restore_reaches_indexed_states(self):
        fill(self.engine, 100)
        lsn1 = self.engine.lsn
        self.engine.update("users", 5, {"age": 0})
        self.engine.delete("users", 6)
        backup_dir = os.path.join(self.root, "backup")
        self.engine.backup(backup_dir)
        self.engine.restore(backup_dir, to_lsn=lsn1)
        self.assertEqual(len(self.engine.index_range("users", "age", 0, 100)), 100)
        self.assertTrue(self.engine.verify()["ok"])
        self.engine.restore(backup_dir)
        ages = [r["id"] for r in self.engine.index_get("users", "age", 0)]
        self.assertIn(5, ages)
        self.assertNotIn(6, [r["id"] for r in self.engine.index_range("users", "age")])
        self.assertTrue(self.engine.verify()["ok"])

    def test_replica_and_pinned_sessions_see_the_same_index(self):
        fill(self.engine, 150)
        lsn1 = self.engine.lsn
        self.engine.update("users", 7, {"age": 0})
        replica_dir = os.path.join(self.root, "replica")
        replica = Replica.create(self.engine, replica_dir)
        session = replica.read_session(lsn1)
        self.assertEqual(len(session.index_range("users", "age", 0, 100)), 150)
        self.assertIn(7, [r["id"] for r in replica.index_get("users", "age", 0)])
        self.assertTrue(replica.verify()["ok"])
        replica.reopen()
        self.assertEqual(len(replica.index_range("users", "age", 0, 100)), 150)
        session.close()
        with self.assertRaises(StorageError):
            replica.insert("users", {"id": 1})
        with self.assertRaises(StorageError):
            replica.create_index("users", "active")

    def test_index_maintenance_adds_no_audit_ops(self):
        fill(self.engine, 50)
        before = len(self.engine.audit())
        self.engine.create_index("users", "active")
        entries = self.engine.audit()
        self.assertEqual(len(entries), before + 1)  # one commit, one audit entry
        self.assertEqual(entries[-1]["ops"], [
            {"op": "create_index", "table": "users", "column": "active"}
        ])


if __name__ == "__main__":
    unittest.main()
