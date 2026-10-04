"""Embedded relational store: catalog, MVCC rows, indexes, transactions, recovery.

Durability model
----------------
Every commit rewrites a complete, compacted image of the database into the page
file through the WAL: chunk pages for the audit log start at page 1, chunk
pages for the state follow, and page 0 is the meta record written last.  The
WAL commit marker therefore makes the whole commit atomic; ``reopen`` (or a
fresh :class:`Engine` on the same directory) replays committed WAL records and
reproduces exactly the committed state.

Concurrency model
-----------------
A single process, single writer engine guarded by one reentrant lock.  Readers
use MVCC snapshot isolation: a transaction reads the newest version that was
committed at or before the horizon it started with, so later commits stay
invisible to it.  Writes are optimistic: touching a row that was written after
the reader's snapshot raises :class:`ConflictError`.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from bisect import bisect_left, insort

from . import query
from .pager import PAGE_SIZE, Pager, StorageError, commit_lsns, read_wal

TYPES = ("int", "text", "bool")
META_PAGE = 0
AUDIT_PAGE = 1
STATE_FORMAT = "kvse-state-1"
DB_FORMAT = "kvse-db-1"
BACKUP_FORMAT = "kvse-backup-1"


class ConflictError(StorageError):
    """Write-write conflict between concurrent snapshot transactions."""


class ReadConflictError(ConflictError):
    """Read-write conflict detected when a serializable transaction commits."""


class ConstraintError(StorageError):
    """Primary key, unique, nullability or type constraint violated."""


ISOLATION_LEVELS = ("snapshot", "serializable")


class Engine:
    """A small relational store persisted in one directory."""

    def __init__(self, root, now_ms=None, page_size=PAGE_SIZE):
        self.root = os.path.abspath(root)
        self.page_size = int(page_size)
        self._now_ms = now_ms
        self._lock = threading.RLock()
        self.pager = Pager(self.root, self.page_size)
        self._reset()
        self._load()

    # ------------------------------------------------------------------ basics
    def _now(self):
        source = self._now_ms
        if source is None:
            return int(time.time() * 1000)
        return int(source() if callable(source) else source)

    def _reset(self):
        self.tables = {}
        self._indexes = {}
        self._audit = []
        self._committed = set()
        self._open = set()
        self._horizon = 0
        self._base_horizon = 0
        self._next_txid = 1
        self.lsn = 0

    def _is_committed(self, txid):
        return txid in self._committed or txid <= self._base_horizon

    def _table(self, name):
        table = self.tables.get(name)
        if table is None:
            raise StorageError("unknown table %s" % name)
        return table

    def has_table(self, name):
        return name in self.tables

    def list_tables(self):
        with self._lock:
            return sorted(self.tables)

    def table_info(self, name):
        with self._lock:
            table = self._table(name)
            return {
                "name": name,
                "columns": [dict(col) for col in table["columns"]],
                "primary_key": table["primary_key"],
                "indexes": dict(table["indexes"]),
                "rows": len(table["rows"]),
            }

    def _reader(self):
        """Pseudo transaction used for engine level reads (txid 0 writes nothing)."""
        return Transaction(self, 0, self._horizon)

    def _coerce_pk(self, table, pk):
        column = [c for c in table["columns"] if c["name"] == table["primary_key"]][0]
        if isinstance(pk, str):
            if column["type"] == "int":
                try:
                    return int(pk)
                except ValueError:
                    raise StorageError("primary key %r is not an int" % pk)
            if column["type"] == "bool":
                if pk.lower() in ("true", "1"):
                    return True
                if pk.lower() in ("false", "0"):
                    return False
                raise StorageError("primary key %r is not a bool" % pk)
        return pk

    # ------------------------------------------------------------------- MVCC
    def _visible(self, versions, txid, horizon, committed=None):
        """Newest version of a row visible to ``txid`` at ``horizon``.

        ``committed`` optionally overrides the live committed set with the
        transactions that were committed when a serializable reader began,
        which keeps its reads pinned to the initial snapshot even when an older
        transaction id commits out of order.
        """
        def is_committed(marker):
            if committed is None:
                return self._is_committed(marker)
            return marker <= self._base_horizon or marker in committed

        best = None
        for version in versions:
            created = version["created"]
            deleted = version["deleted"]
            if created != txid and not (is_committed(created) and created <= horizon):
                continue
            if deleted is None or deleted == txid:
                best = version
            elif not (is_committed(deleted) and deleted <= horizon):
                best = version
        return best

    def _conflicting_writes(self, versions, tx):
        """First version written by another transaction after ``tx`` started."""
        for version in versions:
            for marker in (version["created"], version["deleted"]):
                if marker is None or marker == tx.txid:
                    continue
                if marker > tx.snapshot or not self._is_committed(marker):
                    return version
        return None

    # ------------------------------------------------- serializable conflict
    def _committed_writers(self, versions, tx):
        """Other transactions that committed on this row after ``tx`` began.

        Commit ordering is irrelevant: a transaction with a smaller id that was
        still open when ``tx`` began and commits later is also reported.  The
        frozen committed set captured at begin time is the boundary.
        """
        frozen = tx._committed_at_start
        writers = set()
        for version in versions:
            for marker in (version["created"], version["deleted"]):
                if marker is None or marker == tx.txid or not self._is_committed(marker):
                    continue
                if marker <= self._base_horizon or marker in frozen:
                    continue  # already committed before tx began
                writers.add(marker)
        return writers

    @staticmethod
    def _matches(values, conds):
        """True when a row image satisfies every protected condition."""
        if values is None:
            return False
        for column, op, expected in conds:
            try:
                if not query.compare(op, values.get(column), expected):
                    return False
            except StorageError:
                return False  # incomparable images never satisfy the predicate
        return True

    @staticmethod
    def _write_images(versions, writer):
        """``(before, after)`` net row images for ``writer`` on this row.

        Writes to one row are serialized (a concurrent write is a write-write
        conflict), so ``writer`` owns one contiguous segment of the chain; its
        intermediate versions (insert then update, or an insert later deleted
        inside the same transaction) are collapsed.  ``None`` marks absence:
        an insert is ``(None, new)``, a delete ``(old, None)`` and a row
        inserted then deleted before commit is ``(None, None)``.
        """
        before = None
        for version in versions:
            if version["deleted"] == writer and version["created"] != writer:
                before = version["values"]
                break
        after = None
        for version in versions:
            if version["created"] == writer and version["deleted"] != writer:
                after = version["values"]
                break
        return before, after

    def _check_read_conflict(self, tx):
        """Return a human readable conflict reason, or ``None`` when serializable
        transaction ``tx`` may commit.

        A conflict exists when another transaction that committed after ``tx``
        began wrote a protected primary key (for ``get``), wrote anything in a
        protected table (for ``scan`` or a predicate-less ``query``) or wrote a
        row whose before- or after-image satisfies a protected predicate (index
        reads and ``query`` where clauses).  The other transaction's isolation
        level and commit ordering are irrelevant.
        """
        if tx.isolation != "serializable":
            return None
        for table, protected in tx._protected.items():
            table_obj = self.tables.get(table)
            if table_obj is None:
                continue
            for pk, versions in table_obj["rows"].items():
                writers = self._committed_writers(versions, tx)
                if not writers:
                    continue
                if protected["all"]:
                    return "table %s primary key %r was written by committed transaction %d" % (
                        table, pk, sorted(writers)[0],
                    )
                for other in sorted(writers):
                    before, after = self._write_images(versions, other)
                    for predicate in protected["predicates"]:
                        if self._matches(before, predicate) or self._matches(after, predicate):
                            return (
                                "write by transaction %d on table %s primary key %r "
                                "matches a protected read predicate"
                            ) % (other, table, pk)
        return None

    # ------------------------------------------------------------- read paths
    def _scan(self, tx, table):
        table_obj = self._table(table)
        tx._protect_table(table)
        rows = []
        for pk in sorted(table_obj["rows"], key=query.sort_key):
            version = self._visible(
                table_obj["rows"][pk], tx.txid, tx.snapshot, tx._snapshot_committed
            )
            if version is not None:
                rows.append(dict(version["values"]))
        return rows

    def _get(self, tx, table, pk):
        table_obj = self._table(table)
        pk = self._coerce_pk(table_obj, pk)
        tx._protect_predicate(table, [(table_obj["primary_key"], "=", pk)])
        versions = table_obj["rows"].get(pk)
        if not versions:
            return None
        version = self._visible(versions, tx.txid, tx.snapshot, tx._snapshot_committed)
        return dict(version["values"]) if version is not None else None

    def _index_lookup(self, tx, table, column, low, high):
        """Rows whose current version falls inside [low, high], in index order.

        Stale entries left behind by superseded versions are filtered out by
        checking the value of the version this transaction can actually see.
        """
        table_obj = self._table(table)
        if column not in table_obj["indexes"]:
            raise StorageError("no index on %s.%s" % (table, column))
        bounds = []
        if low is not None:
            bounds.append((column, ">=", low))
        if high is not None:
            bounds.append((column, "<=", high))
        tx._protect_predicate(table, bounds)
        entries = self._indexes.get((table, column), [])
        start = 0 if low is None else bisect_left(entries, (low,))
        rows = []
        seen = set()
        for value, pk in entries[start:]:
            if high is not None and value > high:
                break
            if pk in seen:
                continue
            versions = table_obj["rows"].get(pk)
            version = (
                self._visible(versions, tx.txid, tx.snapshot, tx._snapshot_committed)
                if versions else None
            )
            if version is None:
                continue
            current = version["values"].get(column)
            if current is None or current != value:
                continue  # stale entry of a superseded version
            if (low is not None and current < low) or (high is not None and current > high):
                continue
            seen.add(pk)
            rows.append(dict(version["values"]))
        return rows

    # ------------------------------------------------------------ write paths
    def _read_json(self, page_id):
        payload = self.pager.read_page(page_id).rstrip(b"\x00")
        if not payload:
            raise StorageError("page %d is empty" % page_id)
        return json.loads(payload.decode("utf-8"))

    def _read_pages(self, first_page, count, length):
        chunks = [self.pager.read_page(first_page + i) for i in range(int(count))]
        return b"".join(chunks)[: int(length)]

    def _load(self):
        if self.pager.page_count() == 0:
            return
        meta = self._read_json(META_PAGE)
        if not isinstance(meta, dict) or meta.get("format") != STATE_FORMAT:
            raise StorageError("page 0 does not hold a %s meta record" % STATE_FORMAT)
        self.lsn = int(meta.get("lsn", 0))
        self._horizon = int(meta.get("horizon", 0))
        self._base_horizon = self._horizon
        self._next_txid = max(int(meta.get("next_txid", 1)), self._horizon + 1)
        blob = self._read_pages(meta["state_page"], meta["state_pages"], meta["state_len"])
        state = json.loads(blob.decode("utf-8"))
        if state.get("format") != DB_FORMAT:
            raise StorageError("state blob has an unknown format")
        for name, info in state.get("tables", {}).items():
            table = {
                "name": name,
                "columns": info["columns"],
                "primary_key": info["primary_key"],
                "indexes": {col: "%s_%s_idx" % (name, col) for col in info.get("indexes", [])},
                "rows": {},
            }
            for pk, values in info["rows"]:
                table["rows"][pk] = [{"created": self._horizon, "deleted": None, "values": values}]
            self.tables[name] = table
            self._rebuild_indexes(table)
        audit_blob = self._read_pages(meta["audit_page"], meta["audit_pages"], meta["audit_len"])
        self._audit = json.loads(audit_blob.decode("utf-8")) if audit_blob else []

    def _write_chunks(self, txid, first_page, blob):
        size = self.pager.payload_size
        count = max(1, -(-len(blob) // size))
        for index in range(count):
            self.pager.write_page(txid, first_page + index, blob[index * size:(index + 1) * size])
        return count

    def _snapshot(self):
        tables = {}
        for name, table in self.tables.items():
            rows = []
            for pk in sorted(table["rows"], key=query.sort_key):
                version = self._visible(table["rows"][pk], None, self._horizon)
                if version is not None:
                    rows.append([pk, version["values"]])
            tables[name] = {
                "columns": table["columns"],
                "primary_key": table["primary_key"],
                "indexes": sorted(table["indexes"]),
                "rows": rows,
            }
        return {"format": DB_FORMAT, "tables": tables}

    def _persist(self, tx):
        """Write audit pages, state pages and the meta page, then commit the WAL."""
        audit_lsn = self.pager.peek_lsn()
        entry = {"lsn": audit_lsn, "txid": tx.txid, "at": self._now(), "ops": list(tx.ops)}
        audit = self._audit + [entry]
        audit_blob = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
        audit_pages = self._write_chunks(tx.txid, AUDIT_PAGE, audit_blob)
        state_blob = json.dumps(self._snapshot(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        state_first = AUDIT_PAGE + audit_pages
        state_pages = self._write_chunks(tx.txid, state_first, state_blob)
        meta_lsn = self.pager.peek_lsn()
        meta = {
            "format": STATE_FORMAT,
            "lsn": meta_lsn + 1,
            "txid": tx.txid,
            "audit_page": AUDIT_PAGE,
            "audit_pages": audit_pages,
            "audit_len": len(audit_blob),
            "state_page": state_first,
            "state_pages": state_pages,
            "state_len": len(state_blob),
            "next_txid": self._next_txid,
            "horizon": self._horizon,
        }
        self.pager.write_page(tx.txid, META_PAGE, json.dumps(meta, sort_keys=True).encode("utf-8"))
        lsn = self.pager.commit(tx.txid)
        self._audit = audit
        self.lsn = lsn
        return lsn

    # ------------------------------------------------------------- definition
    def _check_definition(self, name, columns, primary_key, indexes):
        if not isinstance(name, str) or not name.strip():
            raise StorageError("table name must be a non-empty string")
        if not isinstance(columns, list) or not columns:
            raise StorageError("table %s needs at least one column" % name)
        seen = set()
        for column in columns:
            if not isinstance(column, dict):
                raise StorageError("column definitions must be objects")
            cname = column.get("name")
            if not isinstance(cname, str) or not cname.strip():
                raise StorageError("column name must be a non-empty string")
            if cname in seen:
                raise StorageError("duplicate column %s" % cname)
            seen.add(cname)
            if column.get("type") not in TYPES:
                raise StorageError("column %s has unsupported type %r" % (cname, column.get("type")))
            for flag in ("nullable", "unique"):
                if flag in column and not isinstance(column[flag], bool):
                    raise StorageError("column %s: %s must be a boolean" % (cname, flag))
        if primary_key not in seen:
            raise StorageError("primary key %r is not a column of table %s" % (primary_key, name))
        if indexes is None:
            indexes = []
        if not isinstance(indexes, (list, tuple)):
            raise StorageError("indexes must be a list of column names")
        for column in indexes:
            if column not in seen:
                raise StorageError("index column %r is not a column of table %s" % (column, name))

    def create_table(self, name, columns, primary_key, indexes=None):
        with self._lock:
            self._check_definition(name, columns, primary_key, indexes)
            if name in self.tables:
                raise StorageError("table %s already exists" % name)
            tx = self._begin()
            try:
                catalog = []
                for column in columns:
                    catalog.append({
                        "name": column["name"],
                        "type": column["type"],
                        "nullable": False if column["name"] == primary_key else bool(column.get("nullable", True)),
                        "unique": True if column["name"] == primary_key else bool(column.get("unique", False)),
                    })
                self.tables[name] = {
                    "name": name,
                    "columns": catalog,
                    "primary_key": primary_key,
                    "indexes": {},
                    "rows": {},
                }
                tx._undo.append(lambda: self._drop_table(name))
                for column in (indexes or []):
                    self._create_index(tx, name, column)
                tx.ops.append({
                    "op": "create_table",
                    "table": name,
                    "primary_key": primary_key,
                    "indexes": sorted(indexes or []),
                })
                tx.commit()
            except Exception:
                tx.rollback()
                raise
            return self.table_info(name)

    def _drop_table(self, name):
        self.tables.pop(name, None)
        for key in [k for k in self._indexes if k[0] == name]:
            self._indexes.pop(key, None)

    def create_index(self, table, column):
        with self._lock:
            tx = self._begin()
            try:
                name = self._create_index(tx, table, column)
                tx.ops.append({"op": "create_index", "table": table, "column": column})
                tx.commit()
            except Exception:
                tx.rollback()
                raise
            return name

    def _create_index(self, tx, table, column):
        table_obj = self._table(table)
        if column not in {col["name"] for col in table_obj["columns"]}:
            raise StorageError("unknown column %s.%s" % (table, column))
        if column in table_obj["indexes"]:
            raise StorageError("index on %s.%s already exists" % (table, column))
        table_obj["indexes"][column] = "%s_%s_idx" % (table, column)
        self._indexes[(table, column)] = []
        self._rebuild_indexes(table_obj)
        tx._undo.append(lambda: self._drop_index(table, column))
        return table_obj["indexes"][column]

    def _drop_index(self, table, column):
        self._indexes.pop((table, column), None)
        table_obj = self.tables.get(table)
        if table_obj is not None:
            table_obj["indexes"].pop(column, None)

    def _rebuild_indexes(self, table):
        for column in table["indexes"]:
            entries = self._indexes.setdefault((table["name"], column), [])
            del entries[:]
            for pk, versions in table["rows"].items():
                version = self._visible(versions, None, self._horizon)
                if version is None:
                    continue
                value = version["values"].get(column)
                if value is not None:
                    insort(entries, (value, pk))

    def _index_add(self, table, values):
        """Add index entries for one row version; never duplicates.

        Entries of superseded versions are kept until nothing can read them
        any more (see ``_finish``), so an index scan stays correct for readers
        that still hold an older snapshot.
        """
        table_obj = self.tables[table]
        pk = values[table_obj["primary_key"]]
        for column in table_obj["indexes"]:
            value = values.get(column)
            if value is None:
                continue
            entries = self._indexes.setdefault((table, column), [])
            entry = (value, pk)
            if entry not in entries:
                insort(entries, entry)

    # ------------------------------------------------------------- row writes
    def _build_values(self, table, row, current):
        if not isinstance(row, dict):
            raise ConstraintError("row must be a JSON object")
        columns = {col["name"]: col for col in table["columns"]}
        unknown = sorted(key for key in row if key not in columns)
        if unknown:
            raise StorageError("unknown column(s) %s on table %s" % (", ".join(unknown), table["name"]))
        merged = dict(current or {})
        merged.update(row)
        values = {}
        for column in table["columns"]:
            name = column["name"]
            value = merged.get(name)
            if value is None:
                if not column["nullable"]:
                    raise ConstraintError("column %s of %s may not be null" % (name, table["name"]))
                values[name] = None
            elif not query.check_type(value, column["type"]):
                raise ConstraintError(
                    "column %s of %s expects %s" % (name, table["name"], column["type"])
                )
            else:
                values[name] = value
        return values

    def _check_unique(self, tx, table, values, exclude_pk):
        primary_key = table["primary_key"]
        for column in table["columns"]:
            name = column["name"]
            if name == primary_key or not column["unique"]:
                continue
            value = values[name]
            if value is None:
                continue
            for pk, versions in table["rows"].items():
                if pk == exclude_pk:
                    continue
                version = self._visible(versions, tx.txid, tx.snapshot)
                if version is not None and version["values"].get(name) == value:
                    raise ConstraintError("unique constraint violated on %s.%s" % (table["name"], name))
                seen = self._conflicting_writes(versions, tx)
                if seen is not None and seen["values"].get(name) == value and seen["deleted"] is None:
                    raise ConflictError(
                        "concurrent write on unique column %s.%s" % (table["name"], name)
                    )

    def _insert(self, tx, table, row):
        table_obj = self._table(table)
        values = self._build_values(table_obj, row, None)
        pk = values[table_obj["primary_key"]]
        versions = table_obj["rows"].get(pk, [])
        if self._visible(versions, tx.txid, tx.snapshot) is not None:
            raise ConstraintError("duplicate primary key %r in table %s" % (pk, table))
        if self._conflicting_writes(versions, tx) is not None:
            raise ConflictError("primary key %r in table %s is written by another transaction" % (pk, table))
        self._check_unique(tx, table_obj, values, None)
        version = {"created": tx.txid, "deleted": None, "values": values}
        table_obj["rows"].setdefault(pk, []).append(version)
        self._index_add(table, values)

        def undo():
            chain = table_obj["rows"].get(pk)
            if chain and version in chain:
                chain.remove(version)
            if chain is not None and not chain:
                table_obj["rows"].pop(pk, None)

        tx._undo.append(undo)
        tx.ops.append({"op": "insert", "table": table, "pk": pk, "row": values})
        return dict(values)

    def _update(self, tx, table, pk, patch):
        table_obj = self._table(table)
        pk = self._coerce_pk(table_obj, pk)
        if not isinstance(patch, dict) or not patch:
            raise StorageError("update needs a non-empty patch object")
        versions = table_obj["rows"].get(pk)
        version = self._visible(versions, tx.txid, tx.snapshot) if versions else None
        if version is None:
            if versions and self._conflicting_writes(versions, tx) is not None:
                raise ConflictError("row %r of %s was written after transaction %d started" % (pk, table, tx.txid))
            raise StorageError("row %r not found in table %s" % (pk, table))
        if self._conflicting_writes(versions, tx) is not None:
            raise ConflictError("row %r of %s was written after transaction %d started" % (pk, table, tx.txid))
        values = self._build_values(table_obj, patch, version["values"])
        if values[table_obj["primary_key"]] != pk:
            raise ConstraintError("primary key of %s may not be updated" % table)
        self._check_unique(tx, table_obj, values, pk)
        version["deleted"] = tx.txid
        new_version = {"created": tx.txid, "deleted": None, "values": values}
        versions.append(new_version)
        self._index_add(table, values)

        def undo():
            version["deleted"] = None
            if new_version in versions:
                versions.remove(new_version)

        tx._undo.append(undo)
        tx.ops.append({"op": "update", "table": table, "pk": pk, "patch": dict(patch)})
        return dict(values)

    def _delete(self, tx, table, pk):
        table_obj = self._table(table)
        pk = self._coerce_pk(table_obj, pk)
        versions = table_obj["rows"].get(pk)
        version = self._visible(versions, tx.txid, tx.snapshot) if versions else None
        if version is None:
            if versions and self._conflicting_writes(versions, tx) is not None:
                raise ConflictError("row %r of %s was written after transaction %d started" % (pk, table, tx.txid))
            raise StorageError("row %r not found in table %s" % (pk, table))
        if self._conflicting_writes(versions, tx) is not None:
            raise ConflictError("row %r of %s was written after transaction %d started" % (pk, table, tx.txid))
        version["deleted"] = tx.txid

        def undo():
            version["deleted"] = None

        tx._undo.append(undo)
        tx.ops.append({"op": "delete", "table": table, "pk": pk})
        return dict(version["values"])

    # ------------------------------------------------------------ transactions
    def _begin(self, snapshot=None, isolation="snapshot"):
        if not isinstance(isolation, str):
            raise StorageError("isolation must be a string")
        if isolation not in ISOLATION_LEVELS:
            raise StorageError("unsupported isolation %r" % (isolation,))
        if snapshot is not None:
            snapshot = int(snapshot)
            if snapshot < 0 or snapshot > self._horizon:
                raise StorageError("snapshot %d is outside the committed range 0..%d" % (snapshot, self._horizon))
            if isolation == "serializable":
                raise StorageError("serializable transactions cannot be pinned to an older snapshot")
        else:
            snapshot = self._horizon
        txid = self._next_txid
        self._next_txid += 1
        self._open.add(txid)
        return Transaction(self, txid, snapshot, isolation, frozenset(self._committed))

    def begin(self, snapshot=None, isolation="snapshot"):
        """Start a write transaction, optionally pinned to an older snapshot."""
        with self._lock:
            return self._begin(snapshot, isolation)

    def _finish(self, tx, committed):
        """Retire a transaction; the last one out rebuilds the indexes."""
        self._open.discard(tx.txid)
        if committed:
            self._committed.add(tx.txid)
            if tx.txid > self._horizon:
                self._horizon = tx.txid
        if not self._open:
            for name in list(self.tables):
                self._rebuild_indexes(self.tables[name])

    def insert(self, table, row):
        with self._lock:
            tx = self._begin()
            try:
                values = self._insert(tx, table, row)
                tx.commit()
                return values
            except Exception:
                tx.rollback()
                raise

    def update(self, table, pk, patch):
        with self._lock:
            tx = self._begin()
            try:
                values = self._update(tx, table, pk, patch)
                tx.commit()
                return values
            except Exception:
                tx.rollback()
                raise

    def delete(self, table, pk):
        with self._lock:
            tx = self._begin()
            try:
                values = self._delete(tx, table, pk)
                tx.commit()
                return values
            except Exception:
                tx.rollback()
                raise

    def transaction(self, ops, snapshot=None, isolation="snapshot"):
        """Apply a batch of operations in one transaction; roll back on error."""
        with self._lock:
            tx = self._begin(snapshot, isolation)
            try:
                for op in ops:
                    if not isinstance(op, dict):
                        raise StorageError("each op must be a JSON object")
                    kind = op.get("op")
                    if kind == "insert":
                        tx.insert(op["table"], op.get("row") or {})
                    elif kind == "update":
                        tx.update(op["table"], op.get("pk"), op.get("patch") or {})
                    elif kind == "delete":
                        tx.delete(op["table"], op.get("pk"))
                    else:
                        raise StorageError("unsupported op %r" % (kind,))
                lsn = tx.commit()
                return {"committed": True, "lsn": lsn, "txid": tx.txid}
            except Exception:
                if tx.state == "active":
                    tx.rollback()
                raise

    # ------------------------------------------------------------ read helpers
    def get(self, table, pk):
        with self._lock:
            return self._get(self._reader(), table, pk)

    def scan(self, table):
        with self._lock:
            return self._scan(self._reader(), table)

    def index_get(self, table, column, value):
        with self._lock:
            return self._index_lookup(self._reader(), table, column, value, value)

    def index_range(self, table, column, low=None, high=None):
        with self._lock:
            return self._index_lookup(self._reader(), table, column, low, high)

    def query(self, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
        with self._lock:
            return query.execute(
                self, table, columns=columns, where=where, index_hint=index_hint,
                limit=limit, order_by=order_by,
            )

    def explain(self, table, where=None, index_hint=None, **kwargs):
        with self._lock:
            return query.explain(self, table, where=where, index_hint=index_hint, **kwargs)

    def audit(self, limit=None):
        with self._lock:
            entries = [dict(entry) for entry in self._audit]
        if limit is None:
            return entries
        limit = int(limit)
        if limit < 0:
            raise StorageError("limit must be greater than or equal to 0")
        return entries[len(entries) - limit:] if limit else []

    def readonly_view(self):
        return ReadOnlyView(self)

    # -------------------------------------------------- recovery and integrity
    def reopen(self):
        """Simulate a restart: drop memory, replay the WAL, reload the state."""
        with self._lock:
            self.pager = Pager(self.root, self.page_size)
            self._reset()
            self._load()
            return {"tables": sorted(self.tables), "lsn": self.lsn, "pages": self.pager.page_count()}

    def verify(self):
        with self._lock:
            pages = self.pager.page_count()
            crc_ok = True
            for page_id in range(pages):
                try:
                    self.pager.read_page(page_id)
                except StorageError:
                    crc_ok = False
            return {
                "pages": pages,
                "wal_records": self.pager.wal_records(),
                "crc_ok": crc_ok,
                "ok": bool(crc_ok and self._indexes_consistent()),
            }

    def _indexes_consistent(self):
        for (table, column), entries in self._indexes.items():
            table_obj = self.tables.get(table)
            if table_obj is None:
                return False
            for value, pk in entries:
                versions = table_obj["rows"].get(pk)
                if not versions or not any(v["values"].get(column) == value for v in versions):
                    return False
            for pk, versions in table_obj["rows"].items():
                version = self._visible(versions, None, self._horizon)
                if version is None:
                    continue
                value = version["values"].get(column)
                if value is not None and (value, pk) not in entries:
                    return False
        return True

    # ------------------------------------------------------------------ backup
    def backup(self, path):
        """Write a consistent copy: pages + WAL + manifest with the current lsn."""
        with self._lock:
            path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            manifest = {
                "format": BACKUP_FORMAT,
                "lsn": self.lsn,
                "at": self._now(),
                "tables": sorted(self.tables),
                "pages": self.pager.page_count(),
                "wal_records": self.pager.wal_records(),
                "page_size": self.page_size,
            }
            shutil.copyfile(self.pager.data_path, os.path.join(path, "data.pages"))
            shutil.copyfile(self.pager.wal_path, os.path.join(path, "wal.log"))
            with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
            manifest["path"] = path
            return manifest

    def restore(self, path, to_lsn=None):
        """Restore a backup, optionally to the committed state at ``to_lsn``."""
        with self._lock:
            path = os.path.abspath(path)
            manifest_path = os.path.join(path, "manifest.json")
            if not os.path.exists(manifest_path):
                raise StorageError("backup manifest not found in %s" % path)
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if manifest.get("format") != BACKUP_FORMAT:
                raise StorageError("unknown backup format %r" % manifest.get("format"))
            target = int(manifest["lsn"]) if to_lsn is None else int(to_lsn)
            if target > int(manifest["lsn"]):
                raise StorageError("lsn %d is newer than backup lsn %d" % (target, manifest["lsn"]))
            records, _ = read_wal(os.path.join(path, "wal.log"))
            markers = commit_lsns(records)
            if target != int(manifest["lsn"]) and (not markers or target < markers[0]):
                raise StorageError("no committed state at or before lsn %d in this backup" % target)
            shutil.copyfile(os.path.join(path, "data.pages"), self.pager.data_path)
            shutil.copyfile(os.path.join(path, "wal.log"), self.pager.wal_path)
            self.pager = Pager(self.root, self.page_size, auto_replay=False)
            applied = self.pager.replay(max_lsn=target)
            self._reset()
            self._load()
            return {
                "requested_lsn": target,
                "restored_lsn": self.lsn,
                "applied_records": applied,
                "tables": sorted(self.tables),
            }


class Transaction:
    """A transaction; snapshot isolated by default, optionally serializable.

    A serializable transaction additionally records the read set of every
    row operation as a protection predicate (a specific primary key for
    ``get``, an index range bound for the index reads, the whole ``where``
    clause for ``query`` or an entire table for ``scan``).  At commit time
    :meth:`Engine._check_read_conflict` rejects the transaction when another
    already committed transaction wrote a row that matched a protected
    predicate before or after that write.
    """

    def __init__(self, engine, txid, snapshot, isolation="snapshot", committed_at_start=None):
        self.engine = engine
        self.txid = txid
        self.snapshot = snapshot
        self.isolation = isolation
        # Transactions already committed when this one began; a serializable
        # reader pins visibility to this set so later (even out of order)
        # commits never leak into its snapshot.
        self._committed_at_start = committed_at_start
        self.state = "active"
        self.ops = []
        self._undo = []
        # table -> {"all": bool, "predicates": set(tuple((col, op, value), ...))}
        self._protected = {}
        self._suppress_protect = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.state == "active":
            self.rollback() if exc_type else self.commit()
        return False

    def _check_active(self):
        if self.state != "active":
            raise StorageError("transaction %d is %s" % (self.txid, self.state))

    # ------------------------------------------------------- read protection
    @property
    def _serializable(self):
        return self.isolation == "serializable" and self.txid != 0

    @property
    def _snapshot_committed(self):
        """Committed set to pin reads to, or ``None`` for live visibility."""
        return self._committed_at_start if self.isolation == "serializable" else None

    def _protect_table(self, table):
        if not self._serializable or self._suppress_protect:
            return
        protected = self._protected.setdefault(table, {"all": False, "predicates": set()})
        protected["all"] = True

    def _protect_predicate(self, table, conds):
        if not self._serializable or self._suppress_protect:
            return
        protected = self._protected.setdefault(table, {"all": False, "predicates": set()})
        if not protected["all"]:
            protected["predicates"].add(tuple(conds))

    # ------------------------------------------------------------------ writes
    def insert(self, table, row):
        self._check_active()
        with self.engine._lock:
            return self.engine._insert(self, table, row)

    def update(self, table, pk, patch):
        self._check_active()
        with self.engine._lock:
            return self.engine._update(self, table, pk, patch)

    def delete(self, table, pk):
        self._check_active()
        with self.engine._lock:
            return self.engine._delete(self, table, pk)

    # ------------------------------------------------------------------- reads
    def get(self, table, pk):
        self._check_active()
        with self.engine._lock:
            return self.engine._get(self, table, pk)

    def scan(self, table):
        self._check_active()
        with self.engine._lock:
            return self.engine._scan(self, table)

    def index_get(self, table, column, value):
        self._check_active()
        with self.engine._lock:
            return self.engine._index_lookup(self, table, column, value, value)

    def index_range(self, table, column, low=None, high=None):
        self._check_active()
        with self.engine._lock:
            return self.engine._index_lookup(self, table, column, low, high)

    def table_info(self, name):
        return self.engine.table_info(name)

    def query(self, table, columns=None, where=None, index_hint=None, limit=None, order_by=None):
        self._check_active()
        with self.engine._lock:
            if not self._serializable:
                return query.execute(
                    self, table, columns=columns, where=where, index_hint=index_hint,
                    limit=limit, order_by=order_by,
                )
            # Validate first: a failed read must not enlarge the protection set.
            conds = query.normalize_where(self, table, where)
            self._suppress_protect = True
            try:
                result = query.execute(
                    self, table, columns=columns, where=where, index_hint=index_hint,
                    limit=limit, order_by=order_by,
                )
            except Exception:
                self._suppress_protect = False
                raise
            self._suppress_protect = False
            # The whole conjunction is protected; projection, ordering, limit
            # and the chosen access path must not narrow it.
            if conds:
                self._protect_predicate(table, conds)
            else:
                self._protect_table(table)
            return result

    # -------------------------------------------------------------- lifecycle
    def commit(self):
        with self.engine._lock:
            self._check_active()
            if self.txid == 0:
                raise StorageError("read-only snapshots cannot be committed")
            engine = self.engine
            conflict = engine._check_read_conflict(self)
            if conflict is not None:
                self._abort_conflict(conflict)
            engine._committed.add(self.txid)
            if self.txid > engine._horizon:
                engine._horizon = self.txid
            try:
                lsn = engine._persist(self)
            except Exception:
                engine._committed.discard(self.txid)
                self._rollback_locked()
                self.state = "failed"
                raise
            self.state = "committed"
            self._undo = []
            engine._finish(self, True)
            return lsn

    def _abort_conflict(self, detail):
        """Roll back and terminate a serializable transaction that lost a conflict."""
        self._rollback_locked()
        self.engine.pager.abort(self.txid)
        self.state = "failed"
        self.engine._finish(self, False)
        raise ReadConflictError(
            "serializable transaction %d conflicts with a committed write: %s" % (self.txid, detail)
        )

    def rollback(self):
        with self.engine._lock:
            self._check_active()
            if self.txid != 0:
                self._rollback_locked()
                self.engine.pager.abort(self.txid)
            self.state = "rolled_back"
            self.engine._finish(self, False)
            return True

    def _rollback_locked(self):
        while self._undo:
            self._undo.pop()()
        self.ops = []


class ReadOnlyView:
    """Handle returned by :meth:`Engine.readonly_view`; every write is refused."""

    _ALLOWED = (
        "get", "scan", "query", "explain", "index_get", "index_range",
        "table_info", "list_tables", "has_table", "audit", "verify",
    )

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        if name in ReadOnlyView._ALLOWED:
            return getattr(self._engine, name)
        raise StorageError("read-only view rejects %s" % name)
