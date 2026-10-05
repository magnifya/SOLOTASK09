"""Disk-backed read-only replicas with consistent reads.

A replica is a second database directory that tracks a primary (source)
database: :meth:`Replica.create` copies the newest committed state, and
:meth:`Replica.sync` later advances the replica to a chosen commit boundary.
The replica directory holds its own page file, WAL, audit log and index
state; the source directory is only ever read, never written, so a replica
can never modify its primary.

Consistency model
-----------------
Every commit of the primary rewrites a complete, compacted image of the
database through the WAL, so the committed WAL prefix up to any commit
marker reproduces exactly that committed state.  A sync therefore snapshots
the source WAL once at call time (commits that land afterwards stay
invisible to this sync), validates the records up to the target commit lsn,
builds the new page image in a scratch directory inside the replica
directory, and only then swaps ``wal.log`` and ``data.pages`` into place.
Any failure - an invalid source or replica path, an incompatible format, a
page or WAL checksum mismatch, missing commit records, a target beyond the
available history or a target that is not a commit boundary - raises
:class:`StorageError` before anything is swapped, so a failed sync never
changes the replica's data or its applied lsn.  Because the applied lsn is
stored in the replica's own meta page, a restarted replica keeps it and
simply continues from the next commit boundary.

Reads
-----
A replica serves the same read surface as :class:`~kvse.engine.Engine`
(``get``, ``scan``, ``index_get``, ``index_range``, ``query``, ``explain``,
``audit`` and ``verify``) by loading its own directory into a private
engine.  :meth:`Replica.read_session` pins one applied commit lsn in a
private copy, so later syncs stay invisible to the session; asking a
session for an lsn beyond the replica's applied boundary raises
:class:`StorageError`.  Every write or commit operation
(``insert``/``update``/``delete``/``create_table``/``create_index``/
``transaction``/``begin``/``restore``/``checkpoint``/``commit``) raises
:class:`StorageError` without touching the WAL or the audit log.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import struct
import tempfile
import threading

from .engine import Engine
from .pager import (
    DATA_NAME,
    HEADER_SIZE,
    PAGE_SIZE,
    Pager,
    StorageError,
    WAL_NAME,
    crc32,
    read_wal,
)

REPLICA_FORMAT = "kvse-replica-1"
REPLICA_META = "replica.json"


# ------------------------------------------------------------------ wal bits
def _record_lsn(record):
    try:
        return int(record.get("lsn", 0) or 0)
    except (TypeError, ValueError):
        raise StorageError("WAL record carries an invalid lsn: %r" % (record.get("lsn"),))


def _commit_lsns(records):
    """Sorted lsn values of every commit marker in ``records``."""
    return sorted(_record_lsn(record) for record in records if record.get("commit"))


def _infer_page_size(records):
    """Page size of the source, derived from its WAL payloads (or None)."""
    size = None
    for record in records:
        if record.get("page_id") is None:
            continue
        payload = record.get("payload_b64")
        if not isinstance(payload, str):
            raise StorageError("source WAL record %s carries no payload" % record.get("lsn"))
        try:
            block = base64.b64decode(payload)
        except (ValueError, binascii.Error) as exc:
            raise StorageError("source WAL record %s is corrupt: %s" % (record.get("lsn"), exc))
        if size is None:
            size = len(block) + HEADER_SIZE
        elif len(block) + HEADER_SIZE != size:
            raise StorageError("source WAL mixes incompatible page sizes")
    return size


def _check_source_pages(data_path, page_size):
    """Validate every block of the source page file without modifying it."""
    try:
        with open(data_path, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        raise StorageError("cannot read source pages: %s" % exc)
    if len(blob) % page_size:
        raise StorageError(
            "source page file is torn: %d bytes is not a multiple of %d"
            % (len(blob), page_size)
        )
    for offset in range(0, len(blob), page_size):
        block = blob[offset:offset + page_size]
        page_id = offset // page_size
        stored_id, stored_crc = struct.unpack(">II", block[:HEADER_SIZE])
        if stored_id != page_id:
            raise StorageError(
                "source page header mismatch: offset says %d, header says %d"
                % (page_id, stored_id)
            )
        if crc32(block[HEADER_SIZE:]) != stored_crc:
            raise StorageError("source page %d failed its crc32 check" % page_id)


def _read_source(source_root, page_size_hint=None):
    """Snapshot the source's WAL records and validate its pages, read-only."""
    if not os.path.isdir(source_root):
        raise StorageError("source database %s does not exist" % source_root)
    data_path = os.path.join(source_root, DATA_NAME)
    wal_path = os.path.join(source_root, WAL_NAME)
    if not os.path.isfile(data_path) or not os.path.isfile(wal_path):
        raise StorageError("%s does not hold a kvse database" % source_root)
    records, _ = read_wal(wal_path)
    inferred = _infer_page_size(records)
    _check_source_pages(data_path, inferred or page_size_hint or PAGE_SIZE)
    return records, inferred


def _resolve_target(records, target_lsn, applied_lsn):
    """The commit lsn to sync to, or None when there is nothing to do."""
    commits = _commit_lsns(records)
    if target_lsn is None:
        if commits and commits[-1] > applied_lsn:
            return commits[-1]
        return None
    if isinstance(target_lsn, bool):
        raise StorageError("target lsn must be an integer")
    try:
        target = int(target_lsn)
    except (TypeError, ValueError):
        raise StorageError("target lsn must be an integer")
    if target < 0:
        raise StorageError("target lsn must be greater than or equal to 0")
    latest = commits[-1] if commits else 0
    if target > latest:
        raise StorageError(
            "target lsn %d is beyond the available history (latest commit lsn %d)"
            % (target, latest)
        )
    if target not in set(commits):
        raise StorageError(
            "lsn %d is not a commit boundary the source can still provide" % target
        )
    if target < applied_lsn:
        raise StorageError(
            "target lsn %d is behind the replica's applied lsn %d" % (target, applied_lsn)
        )
    if target == applied_lsn:
        return None
    return target


def _build_image(records, page_size, target, workdir):
    """Materialize the committed state at ``target`` inside ``workdir``.

    Writes the WAL prefix, replays it (validating every checksum) and loads
    the resulting image into a fresh engine (validating the on-disk formats).
    """
    kept = [record for record in records if 0 < _record_lsn(record) <= target]
    wal_path = os.path.join(workdir, WAL_NAME)
    with open(wal_path, "wb") as fh:
        for record in kept:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    Pager(workdir, page_size)  # replay validates checksums and builds data.pages
    engine = Engine(workdir, page_size=page_size)  # load validates the formats
    if engine.lsn != target:
        raise StorageError(
            "commit records up to lsn %d are incomplete: reached lsn %d" % (target, engine.lsn)
        )
    return engine


def _swap(workdir, root):
    """Move a freshly built image into place, WAL first.

    The WAL goes first because replay rebuilds the page file from it: even a
    crash between the two renames heals itself on the next open.
    """
    os.replace(os.path.join(workdir, WAL_NAME), os.path.join(root, WAL_NAME))
    os.replace(os.path.join(workdir, DATA_NAME), os.path.join(root, DATA_NAME))


def _write_meta(root, meta):
    path = os.path.join(root, REPLICA_META)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _source_root(source):
    if isinstance(source, Engine):
        return source.root
    if isinstance(source, (str, os.PathLike)):
        return os.path.abspath(str(source))
    raise StorageError("source must be an Engine or a database directory path")


# --------------------------------------------------------------- read facade
class _ReadOnly:
    """Shared read surface of replicas and pinned read sessions."""

    def _require_engine(self):
        raise NotImplementedError

    def get(self, table, pk):
        with self._lock:
            return self._require_engine().get(table, pk)

    def scan(self, table):
        with self._lock:
            return self._require_engine().scan(table)

    def index_get(self, table, column, value):
        with self._lock:
            return self._require_engine().index_get(table, column, value)

    def index_range(self, table, column, low=None, high=None):
        with self._lock:
            return self._require_engine().index_range(table, column, low, high)

    def query(self, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
        with self._lock:
            return self._require_engine().query(
                table, columns=columns, where=where, index_hint=index_hint,
                limit=limit, order_by=order_by,
            )

    def explain(self, table, where=None, index_hint=None, **kwargs):
        with self._lock:
            return self._require_engine().explain(table, where=where, index_hint=index_hint, **kwargs)

    def audit(self, limit=None):
        with self._lock:
            return self._require_engine().audit(limit=limit)

    def verify(self):
        with self._lock:
            return self._require_engine().verify()

    def list_tables(self):
        with self._lock:
            return self._require_engine().list_tables()

    def has_table(self, name):
        with self._lock:
            return self._require_engine().has_table(name)

    def table_info(self, name):
        with self._lock:
            return self._require_engine().table_info(name)

    # ------------------------------------------------- writes are refused
    def _refuse(self, name):
        raise StorageError("a read-only replica rejects %s" % name)

    def insert(self, *args, **kwargs):
        self._refuse("insert")

    def update(self, *args, **kwargs):
        self._refuse("update")

    def delete(self, *args, **kwargs):
        self._refuse("delete")

    def create_table(self, *args, **kwargs):
        self._refuse("create_table")

    def create_index(self, *args, **kwargs):
        self._refuse("create_index")

    def transaction(self, *args, **kwargs):
        self._refuse("transaction")

    def begin(self, *args, **kwargs):
        self._refuse("begin")

    def commit(self, *args, **kwargs):
        self._refuse("commit")

    def restore(self, *args, **kwargs):
        self._refuse("restore")

    def checkpoint(self, *args, **kwargs):
        self._refuse("checkpoint")


class Replica(_ReadOnly):
    """A read-only copy of a primary database, persisted in its own directory."""

    def __init__(self, root):
        self.root = os.path.abspath(str(root))
        self._lock = threading.RLock()
        meta = self._read_meta()
        self._source = meta["source"]
        self._page_size = meta["page_size"]
        for name in (DATA_NAME, WAL_NAME):
            if not os.path.isfile(os.path.join(self.root, name)):
                raise StorageError("replica directory %s is missing %s" % (self.root, name))
        self._engine = Engine(self.root, page_size=self._page_size)

    # --------------------------------------------------------------- basics
    def _read_meta(self):
        path = os.path.join(self.root, REPLICA_META)
        if not os.path.isfile(path):
            raise StorageError("%s is not a kvse replica directory" % self.root)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StorageError("cannot read replica metadata: %s" % exc)
        if not isinstance(meta, dict) or meta.get("format") != REPLICA_FORMAT:
            raise StorageError("replica metadata has an unknown format")
        source = meta.get("source")
        if not isinstance(source, str) or not source:
            raise StorageError("replica metadata does not name a source")
        try:
            page_size = int(meta.get("page_size", PAGE_SIZE))
        except (TypeError, ValueError):
            raise StorageError("replica metadata carries a bad page size")
        return {"source": source, "page_size": page_size}

    def _require_engine(self):
        return self._engine

    @property
    def source(self):
        """Identity (absolute directory) of the primary this replica tracks."""
        return self._source

    @property
    def page_size(self):
        return self._page_size

    @property
    def applied_lsn(self):
        """The commit lsn of the source this replica has applied so far."""
        with self._lock:
            return self._engine.lsn

    def info(self):
        with self._lock:
            return {
                "format": REPLICA_FORMAT,
                "root": self.root,
                "source": self._source,
                "applied_lsn": self._engine.lsn,
                "page_size": self._page_size,
                "tables": self._engine.list_tables(),
            }

    def reopen(self):
        """Reload from the replica directory, keeping the applied lsn."""
        with self._lock:
            meta = self._read_meta()
            self._source = meta["source"]
            self._page_size = meta["page_size"]
            self._engine = Engine(self.root, page_size=self._page_size)
            return self.info()

    # -------------------------------------------------------------- creation
    @classmethod
    def create(cls, source, root, page_size=None):
        """Build the initial replica of ``source`` inside ``root``.

        ``source`` is an :class:`Engine` or a database directory; any existing
        data directory works without migration.  The returned replica exposes
        the source identity as ``source`` and the applied commit lsn as
        ``applied_lsn``.
        """
        source_root = _source_root(source)
        root = os.path.abspath(str(root))
        if root == source_root:
            raise StorageError("a replica must live in its own directory, not the source's")
        records, inferred = _read_source(source_root)
        if os.path.exists(root) and os.listdir(root):
            raise StorageError("replica directory %s is not empty" % root)
        created = not os.path.isdir(root)
        try:
            os.makedirs(root, exist_ok=True)
            size = inferred or (int(page_size) if page_size else PAGE_SIZE)
            commits = _commit_lsns(records)
            target = commits[-1] if commits else 0
            tmp = tempfile.mkdtemp(prefix=".sync-", dir=root)
            try:
                _build_image(records, size, target, tmp)
                _swap(tmp, root)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            _write_meta(root, {
                "format": REPLICA_FORMAT,
                "source": source_root,
                "page_size": size,
            })
        except StorageError:
            if created:
                shutil.rmtree(root, ignore_errors=True)
            raise
        except (OSError, ValueError, TypeError, KeyError, binascii.Error) as exc:
            if created:
                shutil.rmtree(root, ignore_errors=True)
            raise StorageError("cannot create replica in %s: %s" % (root, exc))
        return cls(root)

    # ------------------------------------------------------------------ sync
    def sync(self, target_lsn=None):
        """Advance the replica to a committed state of the source.

        Without ``target_lsn`` the replica catches up to the newest complete
        commit boundary reachable when the call started; commits the source
        finishes afterwards are not part of this sync.  With a target, the
        target must be a committed lsn the source can still provide; records
        are applied in commit order up to that boundary and the visible state
        is swapped in one step.  Any failure raises :class:`StorageError`
        and leaves the replica's data and applied lsn untouched.
        """
        with self._lock:
            previous = self._engine.lsn
            records, inferred = _read_source(self._source, self._page_size)
            if inferred is not None and inferred != self._page_size:
                raise StorageError(
                    "source page size %d is incompatible with the replica's %d"
                    % (inferred, self._page_size)
                )
            target = _resolve_target(records, target_lsn, previous)
            if target is None:
                return self._sync_report(previous, previous, False)
            tmp = tempfile.mkdtemp(prefix=".sync-", dir=self.root)
            try:
                _build_image(records, self._page_size, target, tmp)
                _swap(tmp, self.root)
            except StorageError:
                raise
            except (OSError, ValueError, TypeError, KeyError, binascii.Error) as exc:
                raise StorageError("replica sync failed: %s" % exc)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            self._engine = Engine(self.root, page_size=self._page_size)
            return self._sync_report(previous, target, True)

    def _sync_report(self, previous, target, synced):
        return {
            "source": self._source,
            "previous_lsn": previous,
            "target_lsn": target,
            "applied_lsn": self._engine.lsn,
            "synced": synced,
        }

    # ------------------------------------------------------------- sessions
    def read_session(self, lsn=None):
        """Open a read session pinned to one applied commit lsn.

        The session keeps seeing exactly this commit boundary even after
        later syncs; asking for an lsn beyond the replica's applied boundary
        (or one the replica can no longer reproduce) raises
        :class:`StorageError`.
        """
        with self._lock:
            applied = self._engine.lsn
            if lsn is None:
                pinned = applied
            else:
                if isinstance(lsn, bool):
                    raise StorageError("session lsn must be an integer")
                try:
                    pinned = int(lsn)
                except (TypeError, ValueError):
                    raise StorageError("session lsn must be an integer")
            if pinned < 0 or pinned > applied:
                raise StorageError(
                    "lsn %d is outside the replica's applied range 0..%d" % (pinned, applied)
                )
            records, _ = read_wal(self._engine.pager.wal_path)
            if pinned and pinned not in set(_commit_lsns(records)):
                raise StorageError(
                    "the replica no longer holds a commit boundary at lsn %d" % pinned
                )
            return ReplicaSession(self, pinned, records)


class ReplicaSession(_ReadOnly):
    """A read session pinned to one applied commit boundary of its replica."""

    def __init__(self, replica, lsn, records):
        self._replica = replica
        self.lsn = lsn
        self._lock = threading.RLock()
        self._closed = False
        self._tmp = tempfile.mkdtemp(prefix="kvse-replica-session-")
        try:
            self._engine = _build_image(records, replica.page_size, lsn, self._tmp)
        except Exception:
            self._closed = True
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
            raise

    @property
    def applied_lsn(self):
        return self.lsn

    def _require_engine(self):
        if self._closed or self._engine is None:
            raise StorageError("read session is closed")
        return self._engine

    def close(self):
        with self._lock:
            self._closed = True
            self._engine = None
            tmp, self._tmp = self._tmp, None
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
