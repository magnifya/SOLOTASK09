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
primary/unique/not-null/type constraints, secondary indexes, a small query
subset, crash recovery, auditing, read-only views and snapshot backup with
point-in-time restore.

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
from kvse import Engine, StorageError

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
```

## Layout

```
kvse/__init__.py   public API: Engine, Transaction, StorageError
kvse/pager.py      fixed size page file, crc32 headers, WAL append/replay/checkpoint
kvse/engine.py     catalog, MVCC rows, transactions, indexes, constraints,
                   recovery, backup/restore, audit, read-only view
kvse/query.py      query subset: scan / point get / index range + explain
kvse/http_app.py   ThreadingHTTPServer + create_server(engine, host, port)
kvse/cli.py        serve, create-table, insert, get, query, verify, tx-demo
kvse/__main__.py   python3 -m kvse entry point
tests/             unittest suite for pager, engine and HTTP API
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
page 1+a..meta-1  state chunks: {"format":"kvse-db-1","tables":{name: {
                         "columns","primary_key","indexes","rows":[[pk, values]]}}}
```

All chunks share one transaction and are written before the commit marker, so
recovery never sees half a commit.

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
  an index scan stays correct for older snapshots.
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
| POST | `/v1/tx` | `{"ops":[{"op":"insert\|update\|delete","table","row"\|"pk","patch"}],"snapshot_txid"?,"isolation"?}` | 200 `{"committed":true,"lsn":n,"txid":n}` | 400, 409 |
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
* `tests/test_http.py` - the whole HTTP surface on an ephemeral port,
  including 404/400/409 error shapes and a restart.

## Not implemented yet (next steps for the lane)

B+ tree on-disk indexes with page splits, multi-column and covering indexes,
joins/aggregates/ORDER BY pushdown, constraints beyond the four listed above,
triggers, multi-process concurrency with a redo/undo WAL and lock manager,
fuzzy checkpoints and WAL archiving, consistent read-only replicas,
authentication, privileges and audit retention policies.
