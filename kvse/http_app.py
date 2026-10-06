"""Standard library HTTP front end: JSON over HTTP/1.1.

Routes
------
GET  /healthz                      -> {"ok": true}
POST /v1/tables                    -> 201 catalog info
GET  /v1/tables                    -> 200 {"tables": [...]}
POST /v1/tables/{table}/rows       -> 201 {"inserted": n}
GET  /v1/tables/{table}/rows/{pk}  -> 200 row / 404
POST /v1/query                     -> 200 {"rows": [...], "access": "index"|"scan"}
POST /v1/tx                        -> 200 {"committed": true, "lsn": n} / 409
GET  /v1/verify                    -> 200 integrity report
GET  /v1/audit?limit=n             -> 200 {"entries": [...], "count": n}

Every failure is answered with a JSON body ``{"error": "..."}``.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .engine import ConflictError, ConstraintError, Engine, StorageError

ROW_PATH = re.compile(r"^/v1/tables/([^/]+)/rows(?:/(.*))?$")


def coerce(text, type_name):
    """Convert a path/text value into the declared column type."""
    if type_name == "int":
        try:
            return int(text)
        except (TypeError, ValueError):
            raise StorageError("value %r is not an int" % (text,))
    if type_name == "bool":
        lowered = str(text).lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        raise StorageError("value %r is not a bool" % (text,))
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "kvse/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep CLI stdout clean
        return

    # ---------------------------------------------------------------- plumbing
    def _send(self, status, payload):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, {"error": message})

    def _status_for(self, exc):
        if isinstance(exc, (ConflictError, ConstraintError)):
            return 409
        return 400

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise StorageError("invalid JSON body: %s" % exc)
        if not isinstance(data, dict):
            raise StorageError("request body must be a JSON object")
        return data

    @property
    def engine(self):
        return self.server.engine

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/healthz":
                return self._send(200, {"ok": True})
            if path == "/v1/tables":
                return self._send(200, {"tables": self.engine.list_tables()})
            if path == "/v1/verify":
                return self._send(200, self.engine.verify())
            if path == "/v1/audit":
                limit = parse_qs(parsed.query).get("limit", [None])[0]
                limit = int(limit) if limit not in (None, "") else None
                entries = self.engine.audit(limit=limit)
                return self._send(200, {"entries": entries, "count": len(entries)})
            match = ROW_PATH.match(path)
            if match:
                return self._get_row(unquote(match.group(1)), unquote(match.group(2) or ""))
            return self._error(404, "no route for GET %s" % path)
        except Exception as exc:  # noqa: BLE001 - reported as a JSON error
            return self._error(self._status_for(exc), str(exc))

    def _get_row(self, table, pk_text):
        if not self.engine.has_table(table):
            return self._error(404, "unknown table %s" % table)
        info = self.engine.table_info(table)
        column = [c for c in info["columns"] if c["name"] == info["primary_key"]][0]
        row = self.engine.get(table, coerce(pk_text, column["type"]))
        if row is None:
            return self._error(404, "row %s not found in table %s" % (pk_text, table))
        return self._send(200, row)

    # ------------------------------------------------------------------- POST
    def do_POST(self):
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/v1/tables":
                return self._create_table(self._body())
            if path == "/v1/query":
                return self._query(self._body())
            if path == "/v1/tx":
                return self._tx(self._body())
            match = ROW_PATH.match(path)
            if match and not match.group(2):
                return self._insert_rows(unquote(match.group(1)), self._body())
            return self._error(404, "no route for POST %s" % path)
        except Exception as exc:  # noqa: BLE001 - reported as a JSON error
            return self._error(self._status_for(exc), str(exc))

    def _create_table(self, body):
        name = body.get("name")
        info = self.engine.create_table(
            name, body.get("columns") or [], body.get("primary_key"), body.get("indexes") or []
        )
        return self._send(201, dict(info, index_names=sorted(info["indexes"])))

    def _insert_rows(self, table, body):
        if not self.engine.has_table(table):
            return self._error(404, "unknown table %s" % table)
        rows = body.get("rows")
        if not isinstance(rows, list) or not rows:
            raise StorageError("body must contain a non-empty 'rows' array")
        tx = self.engine.begin()
        try:
            for row in rows:
                tx.insert(table, row)
            lsn = tx.commit()
        except Exception:
            if tx.state == "active":
                tx.rollback()
            raise
        return self._send(201, {"inserted": len(rows), "table": table, "lsn": lsn})

    def _query(self, body):
        table = body.get("table")
        if not self.engine.has_table(table):
            return self._error(404, "unknown table %s" % table)
        result = self.engine.query(
            table,
            columns=body.get("columns"),
            where=body.get("where"),
            index_hint=body.get("index_hint"),
            limit=body.get("limit"),
            order_by=body.get("order_by"),
        )
        return self._send(200, result)

    def _tx(self, body):
        ops = body.get("ops")
        if not isinstance(ops, list) or not ops:
            raise StorageError("body must contain a non-empty 'ops' array")
        result = self.engine.transaction(
            ops, snapshot=body.get("snapshot_txid"), isolation=body.get("isolation")
        )
        return self._send(200, result)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, engine):
        self.engine = engine
        super().__init__(address, Handler)


def create_server(engine, host="127.0.0.1", port=8080):
    """Bind a threading HTTP server for ``engine`` and return it unstarted."""
    if not isinstance(engine, Engine):
        raise StorageError("create_server expects an Engine instance")
    return Server((host, int(port)), engine)
