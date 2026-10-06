"""Command line interface.

Usage
-----
python3 -m kvse [--data-dir DIR] serve [--host H] [--port P]
python3 -m kvse [--data-dir DIR] create-table --file table.json
python3 -m kvse [--data-dir DIR] insert --table T --file rows.json
python3 -m kvse [--data-dir DIR] get --table T --pk K
python3 -m kvse [--data-dir DIR] query --table T --where 'col>=value'
python3 -m kvse [--data-dir DIR] verify
python3 -m kvse [--data-dir DIR] tx-demo

Every command prints a single line of JSON on stdout and exits 0; any failure
prints a single line ``{"error": ...}`` on stderr and exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys

from .engine import ConflictError, Engine
from .http_app import create_server
from .pager import StorageError

OPS_BY_LENGTH = ("!=", "<=", ">=", "=", "<", ">")


def emit(payload, stream=None):
    stream = stream if stream is not None else sys.stdout
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise StorageError("cannot read %s: %s" % (path, exc))
    except ValueError as exc:
        raise StorageError("%s does not contain valid JSON: %s" % (path, exc))


def literal(text):
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def parse_where(text):
    for op in OPS_BY_LENGTH:
        index = text.find(op)
        if index > 0:
            return [(text[:index].strip(), op, literal(text[index + len(op):].strip()))]
    raise StorageError("--where must look like <column><op><value>")


def rows_from(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, dict):
        return [data]
    raise StorageError("--file must hold a row object or an array of rows")


def build_parser():
    parser = argparse.ArgumentParser(prog="kvse", description="embedded relational storage engine")
    parser.add_argument("--data-dir", default="./kvse_data", help="directory with data.pages and wal.log")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    create = commands.add_parser("create-table", help="create a table from a JSON file")
    create.add_argument("--file", required=True)

    insert = commands.add_parser("insert", help="insert rows from a JSON file")
    insert.add_argument("--table", required=True)
    insert.add_argument("--file", required=True)

    get = commands.add_parser("get", help="point get by primary key")
    get.add_argument("--table", required=True)
    get.add_argument("--pk", required=True)

    query = commands.add_parser("query", help="run the query subset")
    query.add_argument("--table", required=True)
    query.add_argument("--where", default=None)
    query.add_argument("--columns", default=None)
    query.add_argument("--limit", type=int, default=None)
    query.add_argument("--order-by", default=None)

    commands.add_parser("verify", help="report pages, WAL records and checksums")
    commands.add_parser("tx-demo", help="demonstrate commit, rollback and a conflict")
    return parser


def cmd_serve(args):
    engine = Engine(args.data_dir)
    server = create_server(engine, args.host, args.port)
    host, port = server.server_address[0], server.server_address[1]
    emit({"serving": True, "host": host, "port": port, "data_dir": engine.root})
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_create_table(args):
    engine = Engine(args.data_dir)
    data = load_json(args.file)
    info = engine.create_table(
        data.get("name"), data.get("columns") or [], data.get("primary_key"), data.get("indexes") or [],
        checks=data.get("checks"),
    )
    emit({"created": info["name"], "primary_key": info["primary_key"],
          "indexes": sorted(info["indexes"]), "columns": [c["name"] for c in info["columns"]]})
    return 0


def cmd_insert(args):
    engine = Engine(args.data_dir)
    rows = rows_from(load_json(args.file))
    tx = engine.begin()
    try:
        for row in rows:
            tx.insert(args.table, row)
        lsn = tx.commit()
    except Exception:
        tx.rollback()
        raise
    emit({"inserted": len(rows), "table": args.table, "lsn": lsn})
    return 0


def cmd_get(args):
    engine = Engine(args.data_dir)
    if not engine.has_table(args.table):
        raise StorageError("unknown table %s" % args.table)
    row = engine.get(args.table, args.pk)
    if row is None:
        raise StorageError("row %s not found in table %s" % (args.pk, args.table))
    emit(row)
    return 0


def cmd_query(args):
    engine = Engine(args.data_dir)
    where = parse_where(args.where) if args.where else None
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    emit(engine.query(args.table, columns=columns, where=where, limit=args.limit, order_by=args.order_by))
    return 0


def cmd_verify(args):
    emit(Engine(args.data_dir).verify())
    return 0


def cmd_tx_demo(args):
    engine = Engine(args.data_dir)
    if not engine.has_table("demo_items"):
        engine.create_table(
            "demo_items",
            [
                {"name": "id", "type": "int", "nullable": False},
                {"name": "label", "type": "text", "nullable": False, "unique": True},
                {"name": "qty", "type": "int"},
            ],
            "id",
            indexes=["label", "qty"],
        )
    report = {}
    tx = engine.begin()
    tx.insert("demo_items", {"id": 1, "label": "alpha", "qty": 5})
    report["commit_lsn"] = tx.commit()

    tx = engine.begin()
    tx.insert("demo_items", {"id": 2, "label": "beta", "qty": 9})
    report["rolled_back"] = tx.rollback()

    reader = engine.begin()
    writer = engine.begin()
    writer.update("demo_items", 1, {"qty": 42})
    report["second_commit_lsn"] = writer.commit()
    report["snapshot_still_sees"] = reader.get("demo_items", 1)
    try:
        reader.update("demo_items", 1, {"qty": 43})
        report["conflict"] = False
    except ConflictError as exc:
        report["conflict"] = True
        report["conflict_error"] = str(exc)
        reader.rollback()
    report["index_range_qty_0_100"] = [r["id"] for r in engine.index_range("demo_items", "qty", 0, 100)]
    report["access"] = engine.explain("demo_items", where=[("qty", ">=", 1)])
    report["audit_txids"] = [entry["txid"] for entry in engine.audit()]
    emit(report)
    return 0


COMMANDS = {
    "serve": cmd_serve,
    "create-table": cmd_create_table,
    "insert": cmd_insert,
    "get": cmd_get,
    "query": cmd_query,
    "verify": cmd_verify,
    "tx-demo": cmd_tx_demo,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except Exception as exc:  # noqa: BLE001 - single line JSON error on stderr
        emit({"error": str(exc)}, sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
