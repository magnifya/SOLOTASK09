"""Tests for the page store, crc32 headers and WAL replay."""

import os
import shutil
import tempfile
import unittest

from kvse.pager import PAGE_SIZE, Pager, StorageError, read_wal


class PagerTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-pager-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_read_roundtrip_and_commit(self):
        pager = Pager(self.root)
        self.assertEqual(pager.write_page(1, 3, b"hello"), 1)
        with self.assertRaises(StorageError):
            pager.read_page(3)  # buffered until the transaction commits
        self.assertEqual(pager.commit(1), 2)
        self.assertEqual(pager.read_page(3).rstrip(b"\x00"), b"hello")
        self.assertEqual(pager.page_count(), 4)
        self.assertEqual(pager.wal_records(), 2)
        self.assertEqual(pager.peek_lsn(), 3)

    def test_abort_drops_buffered_pages(self):
        pager = Pager(self.root)
        pager.write_page(1, 1, b"never")
        pager.abort(1)
        pager.commit(1)  # marker without page writes
        self.assertEqual(pager.page_count(), 0)
        with self.assertRaises(StorageError):
            pager.read_page(1)

    def test_checksum_failure_is_reported(self):
        pager = Pager(self.root)
        pager.write_page(1, 2, b"payload")
        pager.commit(1)
        with open(pager.data_path, "r+b") as handle:
            handle.seek(2 * PAGE_SIZE + 100)
            handle.write(b"X")
        with self.assertRaises(StorageError) as caught:
            pager.read_page(2)
        self.assertIn("crc32", str(caught.exception))

    def test_torn_page_is_rejected(self):
        pager = Pager(self.root)
        pager.write_page(1, 1, b"abc")
        pager.commit(1)
        with open(pager.data_path, "r+b") as handle:
            handle.truncate(PAGE_SIZE + 10)
        with self.assertRaises(StorageError) as caught:
            pager.read_page(1)
        self.assertIn("torn", str(caught.exception))

    def test_payload_larger_than_a_page_is_rejected(self):
        pager = Pager(self.root)
        with self.assertRaises(StorageError):
            pager.write_page(1, 0, b"x" * PAGE_SIZE)

    def test_replay_rebuilds_pages_and_ignores_a_torn_tail(self):
        pager = Pager(self.root)
        pager.write_page(1, 1, b"committed")
        pager.commit(1)
        pager.write_page(2, 1, b"uncommitted")  # never committed
        with open(pager.data_path, "wb"):
            pass  # simulate losing every page
        with open(pager.wal_path, "ab") as handle:
            handle.write(b'{"lsn": 9, "txid": 9, "page_id": 1, "payl')
        reopened = Pager(self.root, auto_replay=False)
        self.assertEqual(reopened.replay(), 1)  # one committed page write
        self.assertEqual(reopened.read_page(1).rstrip(b"\x00"), b"committed")
        records, valid = read_wal(reopened.wal_path)
        self.assertEqual([bool(r.get("commit")) for r in records], [False, True, False])
        self.assertEqual(os.path.getsize(reopened.wal_path), valid)  # tail truncated
        self.assertEqual(reopened.wal_records(), 3)

    def test_replay_honours_max_lsn(self):
        pager = Pager(self.root)
        pager.write_page(1, 1, b"first")
        first = pager.commit(1)
        pager.write_page(2, 1, b"second")
        pager.commit(2)
        with open(pager.data_path, "wb"):
            pass
        reopened = Pager(self.root)
        self.assertEqual(reopened.replay(max_lsn=first), 1)
        self.assertEqual(reopened.read_page(1).rstrip(b"\x00"), b"first")

    def test_checkpoint_compacts_the_wal(self):
        pager = Pager(self.root)
        pager.write_page(1, 1, b"kept")
        pager.commit(1)
        info = pager.checkpoint()
        self.assertEqual(info["wal_records_compacted"], 2)
        self.assertEqual(pager.wal_records(), 0)
        self.assertEqual(Pager(self.root).read_page(1).rstrip(b"\x00"), b"kept")


if __name__ == "__main__":
    unittest.main()
