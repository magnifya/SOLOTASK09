"""Persistent read-only replicas with consistent read sessions.

A :class:`Replica` is an independent copy of a primary :class:`Engine`
directory: it owns its own page file, WAL, audit log and rebuilt indexes under
its replica directory, and the primary never touches it.  A sync ships the
committed state of one commit boundary at a time:

* the source is sampled once while a stable image is observed (under the live
  engine lock when an :class:`Engine` is handed in, or with a re-read stability
  check for a plain directory), so commits landing during a sync can never mix
  into the result;
* the image is rebuilt in a fresh ``gen-gN`` generation directory by replaying
  the source WAL prefix that ends exactly at the target commit marker - the
  same mechanism point-in-time restore uses;
* the new generation is published by atomically replacing ``replica.json``.
  Any failure before that point only discards the unfinished generation, so
  the previous data and ``applied_lsn`` stay byte-for-byte untouched.

``read_session(lsn=None)`` pins reads to one applied commit boundary.  A
session keeps seeing that boundary after later syncs; asking for an lsn beyond
the replica's current boundary (or one that is not a committed boundary) raises
:class:`StorageError`.  Every mutating entry point is refused before touching
the WAL or audit log.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from contextlib import nullcontext

from .engine import STATE_FORMAT, Engine, ReadOnlyView
from .pager import PAGE_SIZE, HEADER_SIZE, StorageError, commit_lsns, crc32

REPLICA_FORMAT = "kvse-replica-1"
META_NAME = "replica.json"


def _source_id(source):
    """Identity of a primary: the real path of its data directory."""
    if isinstance(source, Engine):
        return os.path.realpath(source.root)
    if isinstance(source, (str, os.PathLike)) and str(source):
        return os.path.realpath(os.path.abspath(os.fspath(source)))
    raise StorageError("replica source must be an Engine or a primary data directory")


def _wal_records(blob):
    """Parse the valid prefix of a WAL file held in memory."""
    records = []
    pos = 0
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
    return records


def _image_meta(data, page_size):
    """Validate a copied page image and return its meta record.

    Every page header checksum is verified, the meta record is parsed and the
    audit/state extent it references must be present, so a torn or corrupt
    copy is rejected before it can reach a replica.
    """
    if len(data) == 0:
        return None
    if len(data) % page_size:
        raise StorageError(
            "source page file has %d trailing bytes (torn copy)" % (len(data) % page_size)
        )
    count = len(data) // page_size
    for page_id in range(count):
        block = data[page_id * page_size:(page_id + 1) * page_size]
        stored_id = int.from_bytes(block[:4], "big")
        stored_crc = int.from_bytes(block[4:HEADER_SIZE], "big")
        payload = block[HEADER_SIZE:]
        if stored_id != page_id:
            raise StorageError("source page %d has a mismatched page header" % page_id)
        if crc32(payload) != stored_crc:
            raise StorageError("source page %d failed its crc32 check" % page_id)
    meta_payload = data[HEADER_SIZE:page_size].rstrip(b"\x00")
    try:
        meta = json.loads(meta_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StorageError("source meta page is not valid JSON: %s" % exc)
    if not isinstance(meta, dict) or meta.get("format") != STATE_FORMAT:
        raise StorageError("source page 0 does not hold a %s meta record" % STATE_FORMAT)
    try:
        expected_pages = int(meta["state_page"]) + int(meta["state_pages"])
    except (KeyError, TypeError, ValueError):
        raise StorageError("source meta record is missing its state page extent")
    if count < expected_pages:
        raise StorageError(
            "source page image is short: %d pages, meta references %d" % (count, expected_pages)
        )
    return meta


def _capture_source(source, page_size=None):
    """Return one consistent snapshot of a primary's on-disk files.

    Returns ``(page_size, data, records, image_lsn)`` where ``records`` is the
    parsed valid WAL prefix and ``image_lsn`` is the commit boundary the page
    file holds.  A live :class:`Engine` is sampled under its write lock; a
    directory source is sampled repeatedly until two consecutive samples
    agree, which hides commits happening while the files are being copied.
    """
    if isinstance(source, Engine):
        root = source.root
        page_size = int(source.page_size)
        data_path = source.pager.data_path
        wal_path = source.pager.wal_path
        guard = source._lock
        attempts = 1
    else:
        root = os.fspath(source)
        if not os.path.isdir(root):
            raise StorageError("source directory %s does not exist" % root)
        page_size = int(page_size or PAGE_SIZE)
        data_path = os.path.join(root, "data.pages")
        wal_path = os.path.join(root, "wal.log")
        guard = nullcontext()
        attempts = 10
    if not os.path.isfile(data_path):
        raise StorageError("source data.pages not found in %s" % root)
    if not os.path.isfile(wal_path):
        raise StorageError("source wal.log not found in %s" % root)

    def read_once():
        with open(data_path, "rb") as fh:
            data = fh.read()
        with open(wal_path, "rb") as fh:
            wal = fh.read()
        return data, wal

    with guard:
        data = wal = records = None
        image_lsn = 0
        for _ in range(attempts):
            data_a, wal_a = read_once()
            meta_a = _image_meta(data_a, page_size)
            records_a = _wal_records(wal_a)
            data_b, wal_b = read_once()
            if data_b == data_a and wal_b == wal_a:
                data, wal, records = data_a, wal_a, records_a
                image_lsn = 0 if meta_a is None else int(meta_a.get("lsn", 0))
                break
        if data is None:
            raise StorageError("source %s kept changing while it was being copied" % root)

    return page_size, bytes(data), records, image_lsn


def _wal_prefix(records, target):
    """Serialize the WAL prefix ending at the ``target`` commit marker."""
    out = []
    for record in records:
        out.append(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        out.append(b"\n")
        if record.get("commit") and int(record.get("lsn", 0) or 0) == target:
            return b"".join(out)
    raise StorageError("commit record for target lsn %d is missing in the source WAL" % target)


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _build_generation(gen_dir, data, records, page_size, target, image_lsn):
    """Materialize a replica generation at commit boundary ``target``.

    The directory receives the source page image and either no WAL (when the
    image itself is the target boundary) or the WAL records up to and
    including the target commit marker.  Opening an :class:`Engine` on it
    replays exactly that prefix and verifies every page and WAL checksum.
    Any failure removes the directory.  Returns the opened engine, which
    already holds ``target``.
    """
    os.makedirs(gen_dir)
    try:
        markers = set(commit_lsns(records))
        if target > image_lsn:
            # Only possible after a torn crash on the primary: the page image
            # predates a commit marker.  Replay cannot recover it because the
            # image lacks the pages of the intervening commits.
            raise StorageError(
                "source cannot rebuild lsn %d: its pages only reach %d" % (target, image_lsn)
            )
        if target == image_lsn and target not in markers:
            # Checkpointed primary: the image itself is the target boundary and
            # no WAL history survives.
            wal_blob = b""
        else:
            # Carry the full prefix through the target marker.  Even when the
            # image already holds the target this preserves every earlier
            # boundary, so read sessions can pin to them and a reopened
            # replica can still catch up from the next commit.
            wal_blob = _wal_prefix(records, target)
        with open(os.path.join(gen_dir, "data.pages"), "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        with open(os.path.join(gen_dir, "wal.log"), "wb") as fh:
            fh.write(wal_blob)
            fh.flush()
            os.fsync(fh.fileno())
        engine = Engine(gen_dir, page_size=page_size)
        if engine.lsn != target:
            raise StorageError(
                "source rebuild stopped at lsn %d, expected %d" % (engine.lsn, target)
            )
        report = engine.verify()
        if not report["crc_ok"] or not report["ok"]:
            raise StorageError("rebuilt replica generation at lsn %d failed verification" % target)
        return engine
    except BaseException:
        shutil.rmtree(gen_dir, ignore_errors=True)
        raise


class Replica:
    """A persistent, independently stored read-only copy of an Engine.

    Construction forms:

    * ``Replica(primary, replica_dir)`` / ``Replica.create(primary, dir)`` -
      build the initial replica from an :class:`Engine` or a primary data
      directory;
    * ``Replica(replica_dir)`` / ``Replica.open(replica_dir)`` - reopen a
      previously created replica.
    """

    def __init__(self, source, replica_dir=None):
        if replica_dir is None:
            self._load_existing(source)
        else:
            self.root = os.path.abspath(replica_dir)
            meta_path = os.path.join(self.root, META_NAME)
            if os.path.exists(meta_path):
                self._load_existing(self.root, expect_source=source)
            else:
                self._create_from(source)

    # ------------------------------------------------------------ open/create
    @classmethod
    def create(cls, source, replica_dir):
        """Build the initial replica of ``source`` and return it.

        A replica that already exists in ``replica_dir`` is not rebuilt; open
        it with :meth:`open` (or the two-argument constructor) instead.
        """
        replica_dir = os.path.abspath(replica_dir)
        if os.path.exists(os.path.join(replica_dir, META_NAME)):
            raise StorageError("a replica already exists in %s" % replica_dir)
        return cls(source, replica_dir)

    @classmethod
    def open(cls, replica_dir):
        """Reopen a previously created replica directory."""
        return cls(replica_dir)

    def _read_meta(self):
        path = os.path.join(self.root, META_NAME)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StorageError("cannot read replica meta %s: %s" % (META_NAME, exc))
        if not isinstance(meta, dict) or meta.get("format") != REPLICA_FORMAT:
            raise StorageError("%s does not hold a %s record" % (META_NAME, REPLICA_FORMAT))
        return meta

    def _load_existing(self, replica_dir, expect_source=None):
        self.root = os.path.abspath(os.fspath(replica_dir))
        if not os.path.isdir(self.root):
            raise StorageError("replica directory %s does not exist" % self.root)
        meta = self._read_meta()
        self.source_id = os.path.realpath(str(meta["source"]))
        if expect_source is not None and _source_id(expect_source) != self.source_id:
            raise StorageError(
                "replica %s was created from %s, not from %s"
                % (self.root, self.source_id, _source_id(expect_source))
            )
        self.page_size = int(meta.get("page_size", PAGE_SIZE))
        self.applied_lsn = int(meta.get("applied_lsn", 0))
        self._seq = int(meta.get("seq", 1))
        self._gen = str(meta["gen"])
        self._source = expect_source if isinstance(expect_source, Engine) else self.source_id
        self._lock = threading.RLock()
        self._refs = {}
        self._snaps = {}
        gen_dir = self._gen_path(self._gen)
        if not os.path.isfile(os.path.join(gen_dir, "data.pages")):
            raise StorageError("replica generation %s is missing" % self._gen)
        self._engine = Engine(gen_dir, page_size=self.page_size)
        if self._engine.lsn != self.applied_lsn:
            raise StorageError(
                "replica meta claims applied_lsn %d but pages hold %d"
                % (self.applied_lsn, self._engine.lsn)
            )
        self._sweep()

    def _create_from(self, source):
        if source is None:
            raise StorageError(
                "no replica exists in %s; pass a primary Engine or directory to create one"
                % self.root
            )
        self.source_id = _source_id(source)
        os.makedirs(self.root, exist_ok=True)
        self.root = os.path.abspath(self.root)
        if self.source_id == os.path.realpath(self.root):
            raise StorageError("replica directory must differ from the primary directory")
        self._lock = threading.RLock()
        self._refs = {}
        self._snaps = {}
        self._source = source
        self._seq = 0
        self._gen = None

        page_size, data, records, image_lsn = _capture_source(source)
        self.page_size = page_size
        markers = commit_lsns(records)
        target = max(image_lsn, markers[-1] if markers else 0)

        # Build straight into the final generation directory.  The switch is
        # published by the atomic replica.json replacement below; if we crash
        # before that, the unreferenced generation is swept on next open.
        gen_dir = self._gen_path("gen-g1")
        shutil.rmtree(gen_dir, ignore_errors=True)
        engine = _build_generation(gen_dir, data, records, page_size, target, image_lsn)
        self._seq = 1
        self._gen = "gen-g1"
        _fsync_dir(self.root)
        self._engine = engine
        self.applied_lsn = target
        self._refs = {self._gen: 0}
        self._write_meta()
        self._sweep()

    # ------------------------------------------------------------- bookkeeping
    def _gen_path(self, name):
        return os.path.join(self.root, name)

    def _write_meta(self):
        meta = {
            "format": REPLICA_FORMAT,
            "source": self.source_id,
            "page_size": self.page_size,
            "applied_lsn": self.applied_lsn,
            "gen": self._gen,
            "seq": self._seq,
        }
        tmp = os.path.join(self.root, META_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, os.path.join(self.root, META_NAME))
        _fsync_dir(self.root)

    def _sweep(self):
        """Remove crash leftovers and generations nothing can reach any more."""
        keep = {self._gen}
        keep.update(name for name, count in self._refs.items() if count > 0)
        keep.update(self._snaps.values())
        with os.scandir(self.root) as entries:
            for entry in entries:
                if not entry.is_dir() or not entry.name.startswith(
                    ("gen-g", "stage-", "snap-")
                ):
                    continue
                if entry.name not in keep:
                    shutil.rmtree(entry.path, ignore_errors=True)

    def _drop_ref(self, name):
        count = self._refs.get(name, 0) - 1
        if count > 0:
            self._refs[name] = count
            return
        self._refs.pop(name, None)
        if name != self._gen:
            shutil.rmtree(self._gen_path(name), ignore_errors=True)
            for lsn, snap_name in list(self._snaps.items()):
                if snap_name == name:
                    self._snaps.pop(lsn, None)

    # ------------------------------------------------------------------- sync
    @property
    def root_path(self):
        return self.root

    def info(self):
        """Report the source identity and the applied commit boundary."""
        with self._lock:
            return {
                "replica": self.root,
                "source": self.source_id,
                "applied_lsn": self.applied_lsn,
                "page_size": self.page_size,
                "tables": self._engine.list_tables(),
            }

    def source_lsn(self):
        """Newest committed boundary the source can offer right now."""
        with self._lock:
            _ps, _data, records, image_lsn = self._captured_source()
            markers = commit_lsns(records)
            return max(image_lsn, markers[-1] if markers else 0)

    def _captured_source(self):
        page_size, data, records, image_lsn = _capture_source(self._source, self.page_size)
        if page_size != self.page_size:
            raise StorageError(
                "source page size changed: replica uses %d, source offers %d"
                % (self.page_size, page_size)
            )
        return page_size, data, records, image_lsn

    def sync(self, target_lsn=None):
        """Apply committed source records up to one commit boundary.

        With no target the replica catches up to the newest complete commit
        boundary visible when the call starts.  With an explicit target the
        target must be a committed lsn the source still retains and must not be
        older than the replica's ``applied_lsn``.  The visible state switches
        once, atomically; any failure leaves the old state and applied lsn
        untouched.
        """
        with self._lock:
            _ps, data, records, image_lsn = self._captured_source()
            markers = commit_lsns(records)
            latest = max(image_lsn, markers[-1] if markers else 0)
            if target_lsn is None:
                target = latest
            else:
                if not isinstance(target_lsn, int) or isinstance(target_lsn, bool):
                    raise StorageError("target_lsn must be an integer commit lsn")
                target = int(target_lsn)
                if target < 0:
                    raise StorageError("target_lsn %d is negative" % target)
                if target > latest:
                    raise StorageError(
                        "target lsn %d is beyond the source history ending at %d"
                        % (target, latest)
                    )
                if target != image_lsn and target not in markers:
                    raise StorageError("lsn %d is not a commit boundary" % target)
            old = self.applied_lsn
            if latest < old:
                # The primary truncated its WAL with a checkpoint and then
                # restarted its lsn sequence, so its history can no longer be
                # ordered against this replica; a fresh replica is required.
                raise StorageError(
                    "source history ends at lsn %d but the replica applied %d; "
                    "the source was checkpointed/truncated and cannot be followed"
                    % (latest, old)
                )
            if target < old:
                raise StorageError(
                    "target lsn %d is older than applied_lsn %d" % (target, old)
                )
            if target == old:
                return {
                    "source": self.source_id,
                    "from_lsn": old,
                    "applied_lsn": target,
                    "applied": False,
                    "caught_up": target == latest,
                }

            seq = self._seq + 1
            new_gen = "gen-g%d" % seq
            gen_dir = self._gen_path(new_gen)
            shutil.rmtree(gen_dir, ignore_errors=True)
            engine = _build_generation(
                gen_dir, data, records, self.page_size, target, image_lsn
            )
            _fsync_dir(self.root)

            old_gen = self._gen
            old_engine = self._engine
            self._seq = seq
            self._gen = new_gen
            self._engine = engine
            self.applied_lsn = target
            try:
                self._write_meta()
            except BaseException:
                # Publication failed: the previous generation is still intact
                # on disk and still named in replica.json; roll memory back.
                self._seq = seq - 1
                self._gen = old_gen
                self._engine = old_engine
                self.applied_lsn = old
                shutil.rmtree(gen_dir, ignore_errors=True)
                raise
            if old_gen is not None and self._refs.get(old_gen, 0) == 0:
                shutil.rmtree(self._gen_path(old_gen), ignore_errors=True)
            self._sweep()
            return {
                "source": self.source_id,
                "from_lsn": old,
                "applied_lsn": target,
                "applied": True,
                "caught_up": target == latest,
            }

    def reopen(self):
        """Reload the replica from disk as if the process had restarted."""
        with self._lock:
            self._load_existing(self.root, expect_source=self._source)
            return {
                "source": self.source_id,
                "applied_lsn": self.applied_lsn,
                "tables": self._engine.list_tables(),
                "pages": self._engine.pager.page_count(),
            }

    # ------------------------------------------------------------- read views
    def read_session(self, lsn=None):
        """Pin reads to an applied commit boundary.

        ``lsn=None`` pins to the current ``applied_lsn``.  The session keeps
        serving that boundary after later syncs; an lsn above the applied
        boundary or outside a retained commit boundary raises
        :class:`StorageError`.
        """
        with self._lock:
            if lsn is None:
                self._refs[self._gen] = self._refs.get(self._gen, 0) + 1
                return ReplicaSession(self, self._gen, self._engine, self.applied_lsn)
            if not isinstance(lsn, int) or isinstance(lsn, bool):
                raise StorageError("session lsn must be an integer commit lsn")
            lsn = int(lsn)
            if lsn < 0 or lsn > self.applied_lsn:
                raise StorageError(
                    "session lsn %d is outside the applied range 0..%d"
                    % (lsn, self.applied_lsn)
                )
            if lsn == self.applied_lsn:
                self._refs[self._gen] += 1
                return ReplicaSession(self, self._gen, self._engine, lsn)
            with open(os.path.join(self._gen_path(self._gen), "wal.log"), "rb") as fh:
                records = _wal_records(fh.read())
            if lsn not in commit_lsns(records):
                raise StorageError("lsn %d is not a retained applied commit boundary" % lsn)
            name = self._snaps.get(lsn)
            if name is None:
                name = "snap-%d" % lsn
                snap_dir = self._gen_path(name)
                shutil.rmtree(snap_dir, ignore_errors=True)
                with open(os.path.join(self._gen_path(self._gen), "data.pages"), "rb") as fh:
                    data = fh.read()
                # A pinned generation always carries the whole WAL prefix, so
                # its page image plus a replay to ``lsn`` rebuilds that view.
                engine = _build_generation(
                    snap_dir, data, records, self.page_size, lsn,
                    image_lsn=self.applied_lsn,
                )
                self._snaps[lsn] = name
            else:
                engine = Engine(self._gen_path(name), page_size=self.page_size)
            self._refs[name] = self._refs.get(name, 0) + 1
            return ReplicaSession(self, name, engine, lsn)

    def _close_session(self, session):
        with self._lock:
            if session._closed:
                return
            name = session._gen_name
            session._closed = True
            self._drop_ref(name)

    # -------------------------------------------------------------- read paths
    @property
    def lsn(self):
        return self.applied_lsn

    @property
    def pager(self):
        return self._engine.pager

    def list_tables(self):
        with self._lock:
            return self._engine.list_tables()

    def has_table(self, name):
        with self._lock:
            return self._engine.has_table(name)

    def table_info(self, name):
        with self._lock:
            return self._engine.table_info(name)

    def get(self, table, pk):
        with self._lock:
            return self._engine.get(table, pk)

    def scan(self, table):
        with self._lock:
            return self._engine.scan(table)

    def index_get(self, table, column, value):
        with self._lock:
            return self._engine.index_get(table, column, value)

    def index_range(self, table, column, low=None, high=None):
        with self._lock:
            return self._engine.index_range(table, column, low, high)

    def query(self, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
        with self._lock:
            return self._engine.query(
                table, columns=columns, where=where, index_hint=index_hint,
                limit=limit, order_by=order_by,
            )

    def explain(self, table, where=None, index_hint=None, **kwargs):
        with self._lock:
            return self._engine.explain(table, where=where, index_hint=index_hint, **kwargs)

    def audit(self, limit=None):
        with self._lock:
            return self._engine.audit(limit=limit)

    def verify(self):
        with self._lock:
            return self._engine.verify()

    # --------------------------------------------------------- write refusal
    def _refuse(self, name):
        raise StorageError("read-only replica rejects %s" % name)

    def insert(self, *a, **k):
        self._refuse("insert")

    def update(self, *a, **k):
        self._refuse("update")

    def delete(self, *a, **k):
        self._refuse("delete")

    def create_table(self, *a, **k):
        self._refuse("create_table")

    def create_index(self, *a, **k):
        self._refuse("create_index")

    def transaction(self, *a, **k):
        self._refuse("transaction")

    def begin(self, *a, **k):
        self._refuse("begin")

    def commit(self, *a, **k):
        self._refuse("commit")

    def rollback(self, *a, **k):
        self._refuse("rollback")

    def restore(self, *a, **k):
        self._refuse("restore")

    def readonly_view(self):
        """A :class:`ReadOnlyView` over the replica's current boundary."""
        return ReadOnlyView(self)

    def backup(self, *a, **k):
        self._refuse("backup")


class ReplicaSession:
    """Read handle pinned to one applied commit boundary."""

    def __init__(self, replica, gen_name, engine, pinned_lsn):
        self._replica = replica
        self._gen_name = gen_name
        self._engine = engine
        self.lsn = pinned_lsn
        self.applied_lsn = pinned_lsn
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self._replica._close_session(self)

    def _check_open(self):
        if self._closed:
            raise StorageError("read session at lsn %d is closed" % self.lsn)

    @property
    def root(self):
        return self._engine.root

    def list_tables(self):
        self._check_open()
        return self._engine.list_tables()

    def has_table(self, name):
        self._check_open()
        return self._engine.has_table(name)

    def table_info(self, name):
        self._check_open()
        return self._engine.table_info(name)

    def get(self, table, pk):
        self._check_open()
        return self._engine.get(table, pk)

    def scan(self, table):
        self._check_open()
        return self._engine.scan(table)

    def index_get(self, table, column, value):
        self._check_open()
        return self._engine.index_get(table, column, value)

    def index_range(self, table, column, low=None, high=None):
        self._check_open()
        return self._engine.index_range(table, column, low, high)

    def query(self, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
        self._check_open()
        return self._engine.query(
            table, columns=columns, where=where, index_hint=index_hint,
            limit=limit, order_by=order_by,
        )

    def explain(self, table, where=None, index_hint=None, **kwargs):
        self._check_open()
        return self._engine.explain(table, where=where, index_hint=index_hint, **kwargs)

    def audit(self, limit=None):
        self._check_open()
        return self._engine.audit(limit=limit)

    def verify(self):
        self._check_open()
        return self._engine.verify()

    def _refuse(self, name):
        raise StorageError("read-only replica session at lsn %d rejects %s" % (self.lsn, name))

    def insert(self, *a, **k):
        self._refuse("insert")

    def update(self, *a, **k):
        self._refuse("update")

    def delete(self, *a, **k):
        self._refuse("delete")

    def create_table(self, *a, **k):
        self._refuse("create_table")

    def create_index(self, *a, **k):
        self._refuse("create_index")

    def transaction(self, *a, **k):
        self._refuse("transaction")

    def begin(self, *a, **k):
        self._refuse("begin")

    def commit(self, *a, **k):
        self._refuse("commit")

    def rollback(self, *a, **k):
        self._refuse("rollback")

    def restore(self, *a, **k):
        self._refuse("restore")
