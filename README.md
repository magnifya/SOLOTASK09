# kvse - embedded relational storage engine (seed)

`kvse` is a minimal but real embedded relational storage engine written from
scratch with the Python standard library only.  It is the frozen baseline for a
long running lane whose goal is to grow a relational backend: page storage and
write-ahead logging, transactions and isolation levels, MVCC concurrency
control, B+ tree indexes and a query execution subset, constraints and
triggers, crash recovery and integrity checking, snapshot backup and
point-in-time recovery, read-only replicas with consistent reads, and
privileges plus auditing.

This seed already implements the storage and transaction core of that list:
a checksummed 4096 byte page store, a redo write-ahead log with torn-tail
tolerance, single-writer MVCC transactions with snapshot isolation,
primary/unique/not-null/type constraints, secondary indexes persisted as
paged B+ trees, a small query subset, crash recovery, auditing, read-only
views, snapshot backup with point-in-time restore, and persistent read-only
replicas with consistent, lsn-pinned read sessions.

## Requirements

* Python 3.10 or newer (tested with 3.12 on WSL Ubuntu-24.04).
* Standard library only - no third party packages, no build step, no network.
* ASCII sources; deterministic behaviour (the clock is injectable).

## Run it

```bash
# HTTP API on 127.0.0.1:8080 (note: --data-dir comes BEFORE the subcommand)
python3 -m kvse --data-dir ./kvse_data serve --host 127.0.0.1 --port 8080

# the test suite
python3 -m unittest discover -s tests -v
```

Other CLI commands (every one prints a single line of JSON on stdout, and a
single line `{"error": "..."}` on stderr with exit status 1 on failure):

```bash
python3 -m kvse --data-dir ./kvse_data create-table --file table.json
python3 -m kvse --data-dir ./kvse_data insert --table items --file rows.json
python3 -m kvse --data-dir ./kvse_data get --table items --pk 2
python3 -m kvse --data-dir ./kvse_data query --table items --where 'qty>=15'
python3 -m kvse --data-dir ./kvse_data verify
python3 -m kvse --data-dir ./kvse_data tx-demo
```

`table.json` is `{"name", "columns", "primary_key", "indexes"}`; `rows.json` is
a row object or an array of row objects.

```python
from kvse import Engine, Replica, StorageError

engine = Engine("./kvse_data")
engine.create_table(
    "items",
    [{"name": "id", "type": "int", "nullable": False},
     {"name": "label", "type": "text", "unique": True},
     {"name": "qty", "type": "int"}],
    "id",
    indexes=["qty"],
)
tx = engine.begin()
tx.insert("items", {"id": 1, "label": "alpha", "qty": 5})
tx.commit()                       # or tx.rollback()
engine.query("items", where=[("qty", ">=", 5)], order_by="id")
engine.explain("items", where=[("qty", ">=", 5)])   # {"access": "index", ...}

# a persistent read-only replica living in its own directory
replica = engine.create_replica("./kvse_replica")   # or Replica.create(engine, dir)
# replica.source_id, replica.applied_lsn describe the initial copy
engine.insert("items", {"id": 2, "label": "beta", "qty": 9})
replica.sync()                    # catch up to the latest commit boundary
replica.sync(target_lsn=lsn)      # or stop exactly at one committed lsn
session = replica.read_session()  # pin reads to the current boundary
```

## Layout

```
kvse/__init__.py   public API: Engine, Transaction, StorageError
kvse/pager.py      fixed size page file, crc32 headers, WAL append/replay/checkpoint
kvse/engine.py     catalog, MVCC rows, transactions, indexes, constraints,
                   recovery, backup/restore, audit, read-only view
kvse/replica.py    persistent read-only replicas: directory/engine sources,
                   atomic boundary sync, lsn-pinned read sessions
kvse/query.py      query subset: scan / point get / index range + explain
kvse/http_app.py   ThreadingHTTPServer + create_server(engine, host, port)
kvse/cli.py        serve, create-table, insert, get, query, verify, tx-demo
kvse/__main__.py   python3 -m kvse entry point
tests/             unittest suite for pager, engine, replicas and HTTP API
```

## On-disk format

A database is a directory holding `data.pages` and `wal.log`.

### data.pages

A sequence of fixed size blocks, 4096 bytes each:

```
+----------------+----------------+----------------------------------+
| page_id  (4B)  | crc32    (4B)  | payload                  (4088B) |
| big endian     | big endian     | zero padded on the right         |
+----------------+----------------+----------------------------------+
```

* `crc32` covers the payload only (`binascii.crc32`).
* A read is rejected with `StorageError` when the block is shorter than one
  page (torn write), when the stored page id disagrees with the file offset,
  or when the checksum does not match.
* Page `0` is the meta page.  Every commit rewrites the whole reachable state,
  so the page file is always a consistent (compacted) image of the last
  committed transaction.

```
page 0            meta: {"format":"kvse-state-1","lsn","txid","audit_page",
                         "audit_pages","audit_len","state_page","state_pages",
                         "state_len","next_txid","horizon"}
page 1..1+a-1     audit log chunks: [{"lsn","txid","at","ops":[...]}, ...]
page 1+a..1+a+b-1 index section (only when some table declares an index):
                  B+ tree pages followed by the index directory chunks
page 1+a+b..      state chunks: {"format":"kvse-db-1","tables":{name: {
                         "columns","primary_key","indexes","rows":[[pk, values]]}},
                         "index_section": {"tree_first","tree_count",
                         "dir_page","dir_pages","dir_len"}?}
```

All chunks share one transaction and are written before the commit marker, so
recovery never sees half a commit - table pages and index pages are atomic
together.  Images written before indexes were persisted carry no
`index_section` and open unchanged: their indexes are rebuilt from the row
versions exactly as before, and the next commit starts writing index pages.

### B+ tree index pages

Each secondary index is stored as a B+ tree of fixed size pages, bulk-loaded
from the committed entries at every commit.  A leaf page holds sorted
`[[key, value, pk], ...]` entries plus a `next` page id; an internal page
holds separator `keys` plus `children` page ids, one more child than keys.
`key` is `[sort_key(value), sort_key(pk)]`, so scans are ordered by value and
then by primary key; null values are never indexed.  A page that would
overflow splits into a fresh one, and the directory chunk lists every index
with its root page.  Loading and `verify` check page numbers, checksums,
root-to-leaf connectivity, key order and primary key references, and compare
the trees against the committed rows; a corrupt, missing or mis-sorted page
makes `verify` report `ok: false` and makes opening the directory raise
`StorageError`.

### wal.log

One JSON object per line, either a page write

```json
{"lsn": 7, "txid": 3, "page_id": 2, "payload_b64": "...", "crc32": 1234567}
```

or a commit marker

```json
{"lsn": 9, "txid": 3, "commit": true}
```

`lsn` is a monotonically increasing write-ahead log sequence number.
`write_page` appends a page-write record; the page file is only modified by
`commit`, which appends the marker first and then applies the buffered pages.
`replay()` re-applies the page writes of every transaction that owns a commit
marker - transactions without one are ignored, which makes recovery atomic and
also drives point-in-time restore.  A partial last line (torn write) or an
unparsable tail is ignored and truncated away instead of aborting recovery;
checksum mismatches inside a record are still reported as errors.
`checkpoint()` rewrites `wal.log` empty once the page file is known to hold
every committed page.

## Transactions and isolation

* `Engine.begin(snapshot=None)` returns a `Transaction` with
  `insert/update/delete/get/scan/index_get/index_range/query` plus
  `commit()`/`rollback()`; it also works as a context manager.
* **Snapshot isolation.** A transaction reads the newest row version that was
  committed at or before the transaction id horizon it started with.  A reader
  that began earlier never sees a later commit, and uncommitted versions of
  other transactions are invisible.
* **Single writer.** One reentrant lock (`threading.RLock`) guards the engine,
  so writers are serialized and every statement is atomic.
* **Write-write conflicts.** Writing a row whose newest version was created or
  deleted after the writer's snapshot (including by an uncommitted
  transaction) raises `ConflictError`.  `begin(snapshot=txid)` lets a client
  pin an older snapshot for optimistic concurrency control; the HTTP API
  exposes it as `snapshot_txid`.
* **Serializable isolation.** `Engine.begin(isolation="serializable")` (also
  on `Engine.transaction` and as `isolation` in `POST /v1/tx`; it cannot be
  combined with a pinned snapshot) reads the same snapshot but records every
  read as a predicate: `get` protects one primary key (even a missing one),
  `scan` the whole table, `index_get`/`index_range` the equality or range
  bounds, and `query` the full matching range of all `where` predicates
  regardless of projection, ordering, limit or access path.  At commit time
  the writes of every transaction that committed after the reader started —
  whatever its isolation level or start order — are replayed against those
  predicates (row images before and after each write); a match raises
  `ConflictError`, rolls the transaction back completely and terminates it.
  Writes outside the protected ranges, the transaction's own writes and
  uncommitted or rolled back writes never conflict.
* Rows are version chains (`created`/`deleted` transaction ids).  Rolling back
  reverses the changes and drops the transaction's buffered pages.  Index
  entries of superseded versions are kept while other transactions are open
  and rebuilt from the live versions as soon as the last writer finishes, so
  an index scan stays correct for older snapshots.  The persisted B+ tree
  pages always hold exactly the committed entries: they are written inside
  the same WAL transaction as the table pages, add no audit operations and
  no extra commit boundaries, and are validated against the committed rows
  on every load.
* **Savepoints.** `tx.savepoint(name)` marks a named point inside an active
  transaction, `tx.rollback_to(name)` undoes every write made after it
  (inserts, updates, deletes, including primary key and unique occupancy)
  while the transaction stays alive, and `tx.release_savepoint(name)` forgets
  a savepoint without undoing anything.  Rolling back to a savepoint keeps
  the target and the earlier savepoints and invalidates the later ones;
  releasing removes the target and everything after it.  Names are
  case-sensitive, must be non-blank strings and live only inside their
  transaction; a freed name may be reused.  Savepoints never touch the WAL,
  the lsn or the audit log, serializable read predicates recorded before or
  inside a rolled-back region stay protected, and nothing about them survives
  commit, rollback, restart or restore.  `Engine.transaction` and
  `POST /v1/tx` accept them as `{"op":"savepoint"|"rollback_to"|"release_savepoint","name"}`.
* Durability: a commit is acknowledged only after the WAL commit marker is
  fsynced and the pages are applied.  `reopen()` (or constructing a new
  `Engine` on the same directory) replays the WAL and returns exactly the
  committed state.

## Constraints

| Constraint | Behaviour |
| --- | --- |
| primary key | must be a declared, not-null column; unique among live rows; may not be updated |
| `unique: true` | no two live rows may share a non-null value; enforced on insert and update |
| `nullable: false` | the column must be present and non-null |
| `references: {"table","column"}` | foreign key to another table's primary key of the same type; a non-null value must resolve to a visible parent row on insert/update, a still-referenced parent row cannot be deleted, and both checks are repeated at commit — a violation rolls the whole transaction back |
| type | `int` (bool rejected), `text` (str only), `bool` (bool only) |
| unknown column | rejected on insert, update and in query predicates |
| table/column definition | name, duplicate columns, primary key and index columns are validated at `create_table` |
| read-only view | `readonly_view()` refuses every write |

Violations raise `ConstraintError`, conflicts raise `ConflictError`; both are
subclasses of `StorageError`.

## HTTP API

`create_server(engine, host, port)` returns an unstarted
`ThreadingHTTPServer`; `python3 -m kvse serve` starts it.  Errors are always
`{"error": "..."}` with status 400, 404, 409 or 200 as noted.

| Method | Path | Body / query | Success | Errors |
| --- | --- | --- | --- | --- |
| GET | `/healthz` | - | 200 `{"ok": true}` | - |
| POST | `/v1/tables` | `{"name","columns","primary_key","indexes"?}` | 201 table info | 400 |
| GET | `/v1/tables` | - | 200 `{"tables":[...]}` | - |
| POST | `/v1/tables/{table}/rows` | `{"rows":[...]}` | 201 `{"inserted":n,"lsn":n}` | 400, 404, 409 |
| GET | `/v1/tables/{table}/rows/{pk}` | - | 200 row | 404 |
| POST | `/v1/query` | `{"table","columns"?,"where"?,"index_hint"?,"limit"?,"order_by"?}` | 200 `{"rows":[...],"access":"index"\|"scan"}` | 400, 404 |
| POST | `/v1/tx` | `{"ops":[{"op":"insert\|update\|delete","table","row"\|"pk","patch"} \| {"op":"savepoint\|rollback_to\|release_savepoint","name"}],"snapshot_txid"?,"isolation"?}` | 200 `{"committed":true,"lsn":n,"txid":n}` | 400, 409 |
| GET | `/v1/verify` | - | 200 `{"pages":n,"wal_records":n,"crc_ok":bool,"ok":bool}` | - |
| GET | `/v1/audit?limit=n` | - | 200 `{"entries":[...],"count":n}` | 400 |

`where` is a list of `[column, op, value]` predicates with `op` in
`= != < <= > >=`; it also accepts `{"column","op","value"}` objects.

## Backup, point-in-time restore and audit

```python
manifest = engine.backup("./backup")        # pages + wal + manifest (lsn, at, tables)
engine.restore("./backup")                  # state at manifest["lsn"]
engine.restore("./backup", to_lsn=lsn)      # state as of that commit
```

A backup copies `data.pages`, `wal.log` and a `manifest.json` holding a
monotonically increasing `lsn`.  Because every commit rewrites a complete
state image, replaying the WAL up to any earlier commit marker reproduces that
exact committed state.  This works as long as the WAL has not been truncated by
`checkpoint()`, and restoring to an lsn the backup cannot reach raises
`StorageError` instead of guessing.

Every commit appends `{"lsn","txid","at","ops":[...]}` to the audit log
(`engine.audit(limit=None)`, `GET /v1/audit`).  `at` comes from the injectable
`now_ms` clock (`Engine(root, now_ms=...)`, a callable or a constant), which
makes tests deterministic.  `engine.readonly_view()` returns a handle that
serves reads and refuses writes.

## Read-only replicas

```python
replica = Replica.create(engine, "./replica")    # or Replica.create("./kvse_data", "./replica")
replica.source_id          # real path of the primary directory
replica.applied_lsn        # commit boundary the initial copy stopped at
replica.sync()             # follow to the latest complete commit boundary
replica.sync(target_lsn=lsn)
old = Replica.open("./replica")   # reopen after a restart; applied_lsn persists
session = replica.read_session()  # pin to the current boundary
stale = replica.read_session(lsn) # or to any retained applied boundary
```

A replica is a fully independent database directory: it keeps its own page
file, WAL prefix, audit log and rebuilt index state under
`replica.json` + `gen-gN/` generation directories.  The primary never writes
there, and nothing on the replica can write back to the primary.  Existing
primary directories need no migration; the source may be handed in as a live
`Engine` (copied under the write lock) or as a plain path (copied only once
two consecutive samples agree, so a commit landing mid-copy never mixes into
the result).

* **Sync boundaries.** `sync()` applies committed source records in commit
  order and switches the replica's visible state once, when the new
  generation is fully rebuilt.  Without a target it stops at the newest
  complete commit boundary visible when the call starts; commits that land
  during the sync are not part of its result.  An explicit `target_lsn` must
  be a committed lsn the source still retains, not newer than the source
  history and not older than the replica's `applied_lsn`.
* **Atomic failure.** Invalid source/replica paths, incompatible formats,
  page or WAL checksum failures, a missing commit record, a target beyond the
  available history or a target that is not a commit boundary all raise
  `StorageError`.  The new generation is built in its own directory and
  published by atomically replacing `replica.json`; on any error the staging
  material is discarded and the previous data and `applied_lsn` are
  untouched.  After a restart the applied lsn is read back and the next sync
  continues from the following boundary.
* **History and checkpoints.** A replica carries the WAL prefix it applied,
  so earlier commit boundaries stay available for pinned sessions until the
  primary truncates that history with `checkpoint()`.  If the primary
  checkpoints and restarts so its lsn sequence rewinds past the replica's
  applied lsn, the source can no longer be followed and `sync()` raises
  `StorageError`; create a fresh replica from the new image instead.  Syncing
  to a boundary the source no longer retains is rejected the same way.
* **Read parity.** `get`, `scan`, `index_get`, `index_range`, `query`,
  `explain`, `audit`, `verify`, `table_info`/`list_tables`/`has_table` and
  `readonly_view` behave exactly like the engine at the applied boundary,
  including predicate validation and access path choice.
* **Read sessions.** `read_session(lsn=None)` returns a handle pinned to one
  applied commit boundary; it keeps serving that boundary after later syncs
  (its generation directory is retained until the session closes) and is
  usable as a context manager.  An lsn above the replica's current boundary,
  or one that is not a retained commit boundary, raises `StorageError`.
* **No writes.** `insert`, `update`, `delete`, `create_table`,
  `create_index`, `begin`, `transaction`, `commit`, `rollback`, `restore`
  (and `backup`) raise `StorageError` on both the replica and its sessions,
  before any WAL or audit record could be written.

## Verification and tests

```bash
python3 -m unittest discover -s tests -v
```

* `tests/test_pager.py` - page write/read round trip, checksum failure,
  torn-page rejection, WAL replay after losing the page file, torn WAL tail,
  replay bounded by `max_lsn`, checkpoint.
* `tests/test_engine.py` - definition validation, constraints, commit and
  rollback, snapshot isolation, write-write conflicts, pinned snapshots,
  index maintenance, the query subset and access path choice, recovery after
  a simulated crash, backup and point-in-time restore, audit and read-only
  views.
* `tests/test_replica.py` - initial replica from an engine and from a
  directory, ordered boundary syncs, target validation and all-or-nothing
  failure, applied-lsn persistence and continued catch-up, pinned read
  sessions across syncs, read/query/explain parity, write refusal without
  WAL/audit writes, corrupt page/WAL rejection, checkpointed sources.
* `tests/test_http.py` - the whole HTTP surface on an ephemeral port,
  including 404/400/409 error shapes and a restart.
* `tests/test_savepoints.py` - named savepoints: partial rollback of row and
  index writes, name validation and invalidation, release semantics, lsn and
  audit neutrality, serializable predicate survival, batch and HTTP ops.
* `tests/test_btree.py` - persisted B+ tree indexes: page splits and deep
  trees, key order and bounds, restart/checkpoint/backup/restore/replica
  parity, legacy images without index pages, lsn/audit neutrality, and
  corruption (checksum, ordering, pointers, missing pages, wrong references)
  surfacing as `verify()["ok"] == False` and `StorageError` on reopen.

## Not implemented yet (next steps for the lane)

Multi-column and covering indexes, joins/aggregates/ORDER BY pushdown,
constraints beyond the four listed above, triggers, multi-process
concurrency with a redo/undo WAL and lock manager, fuzzy checkpoints and
WAL archiving, inter-process replica transport (network/streaming fetch
instead of local file copying), authentication, privileges and audit
retention policies.
