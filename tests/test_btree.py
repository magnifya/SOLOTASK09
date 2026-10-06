"""Persisted B+ tree index tests: pages, splits, recovery, corruption."""

import json
import os
import shutil
import struct
import tempfile
import unittest
from unittest import mock

from kvse import Engine, Replica, StorageError
from kvse.engine import BTREE_FORMAT, _entry_key
from kvse.pager import HEADER_SIZE, crc32

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text"},
    {"name": "qty", "type": "int"},
]

PAGE_SIZE = 512


def read_state(root, page_size=PAGE_SIZE):
    """Parse the meta record and state blob straight from the page file."""
    with open(os.path.join(root, "data.pages"), "rb") as fh:
        data = fh.read()

    def page(page_id):
        block = data[page_id * page_size:(page_id + 1) * page_size]
        return block[HEADER_SIZE:]

    meta = json.loads(page(0).rstrip(b"\x00").decode("utf-8"))
    blob = b"".join(page(meta["state_page"] + i) for i in range(meta["state_pages"]))
    return meta, json.loads(blob[: meta["state_len"]].decode("utf-8"))


def write_page_raw(root, page_id, payload, page_size=PAGE_SIZE):
    """Overwrite one page with a valid header and checksum."""
    block = payload.ljust(page_size - HEADER_SIZE, b"\x00")
    with open(os.path.join(root, "data.pages"), "r+b") as fh:
        fh.seek(page_id * page_size)
        fh.write(struct.pack(">II", page_id, crc32(block)) + block)


def flip_byte(root, page_id, page_size=PAGE_SIZE):
    with open(os.path.join(root, "data.pages"), "r+b") as fh:
        fh.seek(page_id * page_size + HEADER_SIZE + 10)
        byte = fh.read(1)
        fh.seek(page_id * page_size + HEADER_SIZE + 10)
        fh.write(bytes([byte[0] ^ 0xFF]))


class BtreeBuildTests(unittest.TestCase):
    """The bulk loader itself, exercised without touching the disk."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-btree-build-")
        self.engine = Engine(self.root, page_size=PAGE_SIZE)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, entries):
        return self.engine._build_btree(("t", "c"), entries, 1)

    def check_tree(self, pages, root, entries):
        """Walk the in-memory node map and assert every B+ tree invariant."""
        payload = self.engine.pager.payload_size
        leaves = []

        def first_key(page_id):
            node = pages[page_id]
            if node["kind"] == "leaf":
                return node["entries"][0][0] if node["entries"] else None
            return first_key(node["children"][0])

        def walk(page_id, depth):
            node = pages[page_id]
            self.assertEqual(node["format"], BTREE_FORMAT)
            self.assertLessEqual(
                len(json.dumps(node, sort_keys=True, separators=(",", ":"))), payload
            )
            if node["kind"] == "leaf":
                leaves.append(page_id)
                keys = [item[0] for item in node["entries"]]
                self.assertEqual(keys, sorted(keys))
                self.assertEqual(len(set(map(json.dumps, keys))), len(keys))
                return
            children = node["children"]
            self.assertEqual(len(children), len(node["keys"]) + 1)
            self.assertGreaterEqual(len(children), 2)
            for pos, child in enumerate(children):
                if pos:
                    self.assertEqual(node["keys"][pos - 1], first_key(child))
                walk(child, depth + 1)

        walk(root, 0)
        # leaves are linked in key order and cover the entries exactly
        for pos, page_id in enumerate(leaves):
            expect = leaves[pos + 1] if pos + 1 < len(leaves) else None
            self.assertEqual(pages[page_id]["next"], expect)
        got = [item for page_id in leaves for item in pages[page_id]["entries"]]
        self.assertEqual(got, [[_entry_key(v, pk), v, pk] for v, pk in entries])
        return leaves

    def test_empty_index_is_a_single_leaf_root(self):
        pages, root, next_page = self.build([])
        self.assertEqual((root, next_page), (1, 2))
        self.assertEqual(pages[1]["kind"], "leaf")
        self.assertEqual(pages[1]["entries"], [])
        self.assertIsNone(pages[1]["next"])

    def test_deep_tree_splits_and_stays_connected(self):
        entries = [(i % 91, i) for i in range(5000)]
        entries.sort(key=lambda item: (item[0], item[1]))
        pages, root, _next = self.build(entries)
        kinds = [node["kind"] for node in pages.values()]
        self.assertGreater(kinds.count("leaf"), 100)  # pages split when full
        self.assertGreater(kinds.count("internal"), 1)
        self.assertEqual(pages[root]["kind"], "internal")
        leaves = self.check_tree(pages, root, entries)
        # depth: root -> internal -> leaves for this volume
        self.assertEqual(pages[pages[leaves[0]]["next"]]["kind"], "leaf")
        depth = 0
        node = pages[root]
        while node["kind"] != "leaf":
            node = pages[node["children"][0]]
            depth += 1
        self.assertGreaterEqual(depth, 2)

    def test_text_keys_order_by_sort_key(self):
        entries = [("k%04d" % i, i) for i in range(700)]
        pages, root, _next = self.build(entries)
        self.check_tree(pages, root, entries)


class BtreeEngineTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-btree-")
        self.engine = Engine(self.root, now_ms=lambda: 1000, page_size=PAGE_SIZE)
        self.engine.create_table("items", COLUMNS, "id", indexes=["qty", "name"])

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def fill(self, count, base=0):
        for i in range(count):
            self.engine.insert(
                "items", {"id": base + i, "name": "n%04d" % (base + i), "qty": (base + i) % 37}
            )

    def section(self):
        _meta, state = read_state(self.root)
        return state.get("index_section")

    # ------------------------------------------------------------- structure
    def test_index_pages_are_persisted_with_the_commit(self):
        self.fill(60)
        section = self.section()
        self.assertIsNotNone(section)
        self.assertGreater(section["tree_count"], 4)  # several pages: splits happened
        meta, _state = read_state(self.root)
        self.assertEqual(section["tree_first"], 1 + meta["audit_pages"])
        self.assertEqual(meta["state_page"], section["dir_page"] + section["dir_pages"])
        engine = Engine(self.root, page_size=PAGE_SIZE)
        root_page = json.loads(
            engine.pager.read_page(section["tree_first"]).rstrip(b"\x00").decode("utf-8")
        )
        self.assertEqual(root_page["format"], BTREE_FORMAT)
        report = self.engine.verify()
        self.assertEqual(sorted(report), ["crc_ok", "ok", "pages", "wal_records"])
        self.assertTrue(report["ok"])

    def test_restart_keeps_index_results(self):
        self.fill(50)
        before_get = self.engine.index_get("items", "qty", 7)
        before_range = self.engine.index_range("items", "qty", 3, 40)
        before_name = self.engine.index_range("items", "name", "n0010", "n0040")
        self.engine.reopen()
        self.assertEqual(self.engine.index_get("items", "qty", 7), before_get)
        self.assertEqual(self.engine.index_range("items", "qty", 3, 40), before_range)
        self.assertEqual(self.engine.index_range("items", "name", "n0010", "n0040"), before_name)
        self.assertTrue(self.engine.verify()["ok"])

    def test_key_order_nulls_bounds_and_empty_ranges(self):
        self.engine.insert("items", {"id": 1, "name": None, "qty": None})
        self.engine.insert("items", {"id": 2, "name": "b", "qty": 10})
        self.engine.insert("items", {"id": 3, "name": "a", "qty": 20})
        self.engine.insert("items", {"id": 4, "name": "a", "qty": 20})
        # null values are not indexed
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "qty")], [2, 3, 4])
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "name")], [3, 4, 2])
        # equality, closed bounds, open bounds, empty range
        self.assertEqual([r["id"] for r in self.engine.index_get("items", "qty", 20)], [3, 4])
        self.assertEqual(self.engine.index_get("items", "qty", 99), [])
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "qty", 10, 20)], [2, 3, 4])
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "qty", None, 10)], [2])
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "qty", 20, None)], [3, 4])
        self.assertEqual(self.engine.index_range("items", "qty", 20, 10), [])
        self.engine.reopen()
        self.assertEqual([r["id"] for r in self.engine.index_range("items", "name")], [3, 4, 2])

    def test_updates_deletes_and_rollback_keep_the_tree_consistent(self):
        self.fill(40)
        for i in range(0, 40, 2):
            self.engine.update("items", i, {"qty": 1000 + i})
        for i in range(1, 40, 3):
            self.engine.delete("items", i)
        self.assertTrue(self.engine.verify()["ok"])
        expected = sorted(
            (1000 + i if i % 2 == 0 else i % 37, i) for i in range(40) if i % 3 != 1
        )
        got = [(r["qty"], r["id"]) for r in self.engine.index_range("items", "qty")]
        self.assertEqual(got, expected)
        self.engine.reopen()
        self.assertEqual(
            [(r["qty"], r["id"]) for r in self.engine.index_range("items", "qty")], expected
        )
        self.assertTrue(self.engine.verify()["ok"])

        tx = self.engine.begin()
        tx.insert("items", {"id": 500, "name": "x", "qty": 5})
        tx.delete("items", 0)
        tx.rollback()
        self.assertEqual(self.engine.index_get("items", "name", "x"), [])
        self.assertIsNotNone(self.engine.get("items", 0))
        self.engine.reopen()
        self.assertEqual(self.engine.index_get("items", "name", "x"), [])
        self.assertTrue(self.engine.verify()["ok"])

    def test_savepoint_rollback_restores_index_visibility(self):
        self.fill(20)
        tx = self.engine.begin()
        tx.savepoint("sp")
        tx.insert("items", {"id": 900, "name": "sp-row", "qty": 3})
        tx.update("items", 0, {"qty": 777})
        self.assertEqual(tx.index_get("items", "name", "sp-row")[0]["id"], 900)
        tx.rollback_to("sp")
        tx.commit()
        self.assertEqual(self.engine.index_get("items", "name", "sp-row"), [])
        self.assertNotIn(777, [r["qty"] for r in self.engine.index_range("items", "qty")])
        self.engine.reopen()
        self.assertEqual(self.engine.index_get("items", "name", "sp-row"), [])
        self.assertTrue(self.engine.verify()["ok"])

    def test_no_extra_audit_ops_or_commit_markers(self):
        self.fill(5)

        def commit_markers():
            with open(self.engine.pager.wal_path, "rb") as fh:
                return sum(1 for line in fh if b'"commit":true' in line)

        markers_before = commit_markers()
        audits_before = len(self.engine.audit())
        self.engine.insert("items", {"id": 100, "name": "z", "qty": 1})
        self.assertEqual(commit_markers() - markers_before, 1)  # one commit boundary only
        entries = self.engine.audit()
        self.assertEqual(len(entries) - audits_before, 1)
        self.assertEqual([op["op"] for op in entries[-1]["ops"]], ["insert"])

    # ---------------------------------------------------------- legacy images
    def test_legacy_image_without_index_pages_opens_and_upgrades(self):
        legacy = os.path.join(self.root, "legacy")
        with mock.patch.object(Engine, "_build_index_section", lambda self, first_page: None):
            old = Engine(legacy, now_ms=lambda: 1000, page_size=PAGE_SIZE)
            old.create_table("items", COLUMNS, "id", indexes=["qty"])
            for i in range(30):
                old.insert("items", {"id": i, "name": "n%d" % i, "qty": i % 7})
        _meta, state = read_state(legacy)
        self.assertNotIn("index_section", state)  # really the old layout

        engine = Engine(legacy, page_size=PAGE_SIZE)
        self.assertEqual(
            [r["id"] for r in engine.index_range("items", "qty", 0, 6)],
            [r["id"] for r in old.index_range("items", "qty", 0, 6)],
        )
        self.assertTrue(engine.verify()["ok"])
        engine.insert("items", {"id": 50, "name": "n50", "qty": 1})
        self.assertIsNotNone(read_state(legacy)[1].get("index_section"))
        engine.reopen()
        self.assertEqual(
            [r["id"] for r in engine.index_get("items", "qty", 1)], [1, 8, 15, 22, 29, 50]
        )
        self.assertTrue(engine.verify()["ok"])

    # ------------------------------------------------- checkpoint and backup
    def test_checkpoint_backup_and_restore_keep_index_results(self):
        self.fill(30)
        lsn1 = self.engine.lsn
        first = self.engine.index_range("items", "qty", 0, 36)
        for i in range(20):
            self.engine.insert("items", {"id": 1000 + i, "name": "m%d" % i, "qty": 100 + i})
        lsn2 = self.engine.lsn
        backup = os.path.join(self.root, "backup")
        self.engine.backup(backup)
        self.engine.pager.checkpoint()
        self.engine.reopen()
        self.assertEqual(self.engine.index_range("items", "qty", 0, 36), first)
        self.assertEqual(len(self.engine.index_range("items", "qty", 100, 120)), 20)
        self.assertTrue(self.engine.verify()["ok"])

        self.engine.restore(backup, to_lsn=lsn1)
        self.assertEqual(self.engine.index_range("items", "qty", 0, 36), first)
        self.assertEqual(self.engine.index_range("items", "qty", 100, 120), [])
        self.assertTrue(self.engine.verify()["ok"])
        self.engine.restore(backup, to_lsn=lsn2)
        self.assertEqual(len(self.engine.index_range("items", "qty", 0, 120)), 50)
        self.assertTrue(self.engine.verify()["ok"])

    # ---------------------------------------------------------------- replica
    def test_replica_sync_and_pinned_sessions_match(self):
        self.fill(25)
        lsn1 = self.engine.lsn
        replica = Replica.create(self.engine, os.path.join(self.root, "replica"))
        self.fill(15, base=500)
        replica.sync()
        self.assertEqual(
            replica.index_range("items", "qty", 0, 36), self.engine.index_range("items", "qty", 0, 36)
        )
        self.assertTrue(replica.verify()["ok"])
        with replica.read_session(lsn1) as session:
            self.assertEqual(len(session.index_range("items", "qty", 0, 36)), 25)
            self.assertTrue(session.verify()["ok"])
        self.assertEqual(len(replica.index_range("items", "qty", 0, 36)), 40)
        for call in (lambda: replica.create_index("items", "id"),
                     lambda: replica.insert("items", {"id": 1}),
                     lambda: replica.read_session(lsn1).insert("items", {"id": 1})):
            with self.assertRaises(StorageError):
                call()

    # ------------------------------------------------------------- corruption
    def rewrite_node(self, page_id, node):
        write_page_raw(
            self.root, page_id,
            json.dumps(node, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )

    def read_node(self, page_id):
        engine = Engine(self.root, page_size=PAGE_SIZE)
        return json.loads(engine.pager.read_page(page_id).rstrip(b"\x00").decode("utf-8"))

    def test_corrupt_index_page_fails_verify_and_reopen(self):
        self.fill(30)
        section = self.section()
        self.engine.pager.checkpoint()  # no WAL left to repair the page file
        flip_byte(self.root, section["tree_first"])
        report = self.engine.verify()
        self.assertFalse(report["crc_ok"])
        self.assertFalse(report["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root, page_size=PAGE_SIZE)

    def test_unsorted_leaf_fails_verify_and_reopen(self):
        self.fill(30)
        section = self.section()
        self.engine.pager.checkpoint()
        leaf = self.read_node(section["tree_first"])
        self.assertEqual(leaf["kind"], "leaf")
        leaf["entries"] = list(reversed(leaf["entries"]))
        self.rewrite_node(section["tree_first"], leaf)
        report = self.engine.verify()
        self.assertTrue(report["crc_ok"])  # checksums are fine, the tree is not
        self.assertFalse(report["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root, page_size=PAGE_SIZE)

    def test_out_of_range_pointer_fails_verify_and_reopen(self):
        self.fill(30)
        section = self.section()
        self.engine.pager.checkpoint()
        leaf = self.read_node(section["tree_first"])
        leaf["next"] = 10 ** 6
        self.rewrite_node(section["tree_first"], leaf)
        self.assertFalse(self.engine.verify()["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root, page_size=PAGE_SIZE)

    def test_missing_index_page_fails_verify_and_reopen(self):
        self.fill(30)
        self.engine.pager.checkpoint()
        meta, _state = read_state(self.root)
        last = meta["state_page"] + meta["state_pages"]
        with open(os.path.join(self.root, "data.pages"), "r+b") as fh:
            fh.truncate((last - 1) * PAGE_SIZE)
        self.assertFalse(self.engine.verify()["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root, page_size=PAGE_SIZE)

    def test_wrong_pk_reference_fails_verify(self):
        self.fill(15)
        section = self.section()
        self.engine.pager.checkpoint()
        leaf = self.read_node(section["tree_first"])
        key, value, _pk = leaf["entries"][0]
        leaf["entries"][0] = [key, value, 99999]
        self.rewrite_node(section["tree_first"], leaf)
        self.assertFalse(self.engine.verify()["ok"])
        with self.assertRaises(StorageError):
            Engine(self.root, page_size=PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
