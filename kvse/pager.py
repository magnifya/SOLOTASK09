"""Page store plus write-ahead log for the kvse storage engine.

On-disk format
--------------
data.pages
    Sequence of fixed size blocks (default 4096 bytes).  Every block is
    ``page_id:4B big endian`` + ``crc32(payload):4B big endian`` + payload.
    A block whose header disagrees with its offset, whose checksum does not
    match, or which is shorter than one full page is rejected as torn/corrupt.

wal.log
    One JSON object per line, either a page write
    ``{"lsn","txid","page_id","payload_b64","crc32"}`` or a commit marker
    ``{"lsn","txid","commit":true}``.  Page writes are redo-applied only for
    transactions that own a commit marker, which keeps recovery atomic.  A
    partial (torn) trailing line is ignored and truncated away instead of
    being treated as a fatal error.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import struct
import threading

PAGE_SIZE = 4096
HEADER_SIZE = 8
DATA_NAME = "data.pages"
WAL_NAME = "wal.log"


class StorageError(Exception):
    """Storage level failure: I/O, checksum, bad definition or bad usage."""


def crc32(data):
    return binascii.crc32(data) & 0xFFFFFFFF


def read_wal(path):
    """Return ``(records, valid_bytes)`` for the valid prefix of a WAL file.

    Reading stops at the first incomplete or unparsable line; ``valid_bytes``
    is the offset right after the last whole, parsable record.
    """
    if not os.path.exists(path):
        return [], 0
    with open(path, "rb") as fh:
        blob = fh.read()
    records = []
    pos = 0
    valid = 0
    while True:
        newline = blob.find(b"\n", pos)
        if newline < 0:
            break
        line = blob[pos:newline].strip()
        pos = newline + 1
        if line:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                break
            if not isinstance(record, dict):
                break
            records.append(record)
        valid = pos
    return records, valid


def commit_lsns(records):
    """Sorted lsn values of every commit marker in ``records``."""
    return sorted(int(r.get("lsn", 0)) for r in records if r.get("commit"))


def _encode(page_id, block):
    return struct.pack(">II", page_id, crc32(block)) + block


class Pager:
    """Fixed size page file guarded by crc32 headers and a redo WAL."""

    def __init__(self, root, page_size=PAGE_SIZE, auto_replay=True):
        if int(page_size) <= 2 * HEADER_SIZE:
            raise StorageError("page size %r is too small" % (page_size,))
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.page_size = int(page_size)
        self.payload_size = self.page_size - HEADER_SIZE
        self.data_path = os.path.join(self.root, DATA_NAME)
        self.wal_path = os.path.join(self.root, WAL_NAME)
        self._lock = threading.RLock()
        self._pending = {}
        self._lsn = 0
        for path in (self.data_path, self.wal_path):
            if not os.path.exists(path):
                with open(path, "ab"):
                    pass
        if auto_replay:
            self.replay()

    # ------------------------------------------------------------- log helpers
    def _append(self, record):
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with open(self.wal_path, "ab") as fh:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())

    def peek_lsn(self):
        """The lsn the next WAL record will receive."""
        with self._lock:
            return self._lsn + 1

    def wal_records(self):
        with self._lock:
            return len(read_wal(self.wal_path)[0])

    def page_count(self):
        with self._lock:
            try:
                size = os.path.getsize(self.data_path)
            except OSError:
                return 0
        return size // self.page_size

    # ------------------------------------------------------------- page access
    def write_page(self, txid, page_id, payload):
        """Append a WAL record for one page write; the page is applied on commit."""
        if not isinstance(payload, (bytes, bytearray)):
            raise StorageError("page payload must be bytes")
        if page_id < 0:
            raise StorageError("negative page id %r" % (page_id,))
        if len(payload) > self.payload_size:
            raise StorageError(
                "page payload of %d bytes exceeds %d" % (len(payload), self.payload_size)
            )
        block = bytes(payload).ljust(self.payload_size, b"\x00")
        with self._lock:
            record = {
                "lsn": self._lsn + 1,
                "txid": txid,
                "page_id": int(page_id),
                "payload_b64": base64.b64encode(block).decode("ascii"),
                "crc32": crc32(block),
            }
            self._append(record)
            self._lsn = record["lsn"]
            self._pending.setdefault(txid, {})[int(page_id)] = _encode(int(page_id), block)
            return record["lsn"]

    def commit(self, txid):
        """Write the commit marker, then apply this transaction's pages."""
        with self._lock:
            record = {"lsn": self._lsn + 1, "txid": txid, "commit": True}
            self._append(record)
            self._lsn = record["lsn"]
            self._apply(self._pending.pop(txid, {}))
            return record["lsn"]

    def abort(self, txid):
        """Drop buffered page writes; replay ignores records without a marker."""
        with self._lock:
            self._pending.pop(txid, None)

    def _apply(self, pages):
        if not pages:
            return 0
        with open(self.data_path, "r+b") as fh:
            for page_id in sorted(pages):
                fh.seek(page_id * self.page_size)
                fh.write(pages[page_id])
            fh.truncate((max(pages) + 1) * self.page_size)
            fh.flush()
            os.fsync(fh.fileno())
        return len(pages)

    def read_page(self, page_id):
        if page_id < 0:
            raise StorageError("negative page id %r" % (page_id,))
        with self._lock:
            with open(self.data_path, "rb") as fh:
                fh.seek(page_id * self.page_size)
                block = fh.read(self.page_size)
        if not block:
            raise StorageError("page %d is not present" % page_id)
        if len(block) < self.page_size:
            raise StorageError(
                "page %d is torn: short read of %d bytes" % (page_id, len(block))
            )
        stored_id, stored_crc = struct.unpack(">II", block[:HEADER_SIZE])
        if stored_id != page_id:
            raise StorageError(
                "page header mismatch: offset says %d, header says %d" % (page_id, stored_id)
            )
        payload = block[HEADER_SIZE:]
        if crc32(payload) != stored_crc:
            raise StorageError("page %d failed its crc32 check" % page_id)
        return payload

    # --------------------------------------------------------------- recovery
    def replay(self, max_lsn=None):
        """Re-apply committed WAL records and truncate a torn tail.

        Only transactions whose commit marker is present (and, when
        ``max_lsn`` is given, not newer than that lsn) are applied, so replay
        is atomic per transaction and also drives point-in-time recovery.
        """
        with self._lock:
            records, valid = read_wal(self.wal_path)
            try:
                size = os.path.getsize(self.wal_path)
            except OSError:
                size = 0
            if valid < size:
                with open(self.wal_path, "r+b") as fh:
                    fh.truncate(valid)
                    fh.flush()
                    os.fsync(fh.fileno())
            for record in records:
                self._lsn = max(self._lsn, int(record.get("lsn", 0) or 0))
            allowed = set()
            for record in records:
                if not record.get("commit"):
                    continue
                if max_lsn is None or int(record.get("lsn", 0)) <= max_lsn:
                    allowed.add(record.get("txid"))
            pages = {}
            for record in records:
                if record.get("page_id") is None or record.get("txid") not in allowed:
                    continue
                try:
                    block = base64.b64decode(record["payload_b64"])
                except (KeyError, ValueError, binascii.Error) as exc:
                    raise StorageError("corrupt WAL record at lsn %s: %s" % (record.get("lsn"), exc))
                if len(block) != self.payload_size:
                    raise StorageError(
                        "WAL record %s carries %d payload bytes, expected %d"
                        % (record.get("lsn"), len(block), self.payload_size)
                    )
                if crc32(block) != int(record.get("crc32", -1)):
                    raise StorageError("WAL record %s failed its crc32 check" % record.get("lsn"))
                page_id = int(record["page_id"])
                pages[page_id] = _encode(page_id, block)
            return self._apply(pages)

    def checkpoint(self):
        """Compact the WAL: the page file already holds every committed page."""
        with self._lock:
            if self._pending:
                raise StorageError("cannot checkpoint with uncommitted page writes")
            compacted = len(read_wal(self.wal_path)[0])
            with open(self.wal_path, "wb") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            return {"wal_records_compacted": compacted, "lsn": self._lsn, "pages": self.page_count()}
