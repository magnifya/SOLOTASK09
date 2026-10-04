"""HTTP API tests: a real ThreadingHTTPServer on an ephemeral port."""

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from kvse.engine import Engine
from kvse.http_app import create_server

COLUMNS = [
    {"name": "id", "type": "int", "nullable": False},
    {"name": "name", "type": "text", "nullable": False, "unique": True},
    {"name": "age", "type": "int"},
]


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kvse-http-")
        self.engine = Engine(self.root, now_ms=lambda: 7000)
        self.server = create_server(self.engine, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.root, ignore_errors=True)

    def call(self, method, path, body=None):
        request = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=None if body is None else json.dumps(body).encode("utf-8"),
            method=method,
        )
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def create(self):
        status, body = self.call("POST", "/v1/tables", {
            "name": "users", "columns": COLUMNS, "primary_key": "id", "indexes": ["age"],
        })
        self.assertEqual(status, 201)
        return body

    def test_healthz_and_table_listing(self):
        status, body = self.call("GET", "/healthz")
        self.assertEqual((status, body), (200, {"ok": True}))
        self.assertEqual(self.call("GET", "/v1/tables"), (200, {"tables": []}))
        self.create()
        self.assertEqual(self.call("GET", "/v1/tables"), (200, {"tables": ["users"]}))
        status, body = self.call("POST", "/v1/tables", {
            "name": "users", "columns": COLUMNS, "primary_key": "id",
        })
        self.assertEqual(status, 400)
        self.assertIn("already exists", body["error"])

    def test_rows_point_get_and_query(self):
        self.create()
        status, body = self.call("POST", "/v1/tables/users/rows", {"rows": [
            {"id": 1, "name": "ann", "age": 30},
            {"id": 2, "name": "bob", "age": 20},
            {"id": 3, "name": "cid", "age": 40},
        ]})
        self.assertEqual((status, body["inserted"]), (201, 3))

        status, row = self.call("GET", "/v1/tables/users/rows/2")
        self.assertEqual(status, 200)
        self.assertEqual(row, {"id": 2, "name": "bob", "age": 20})

        status, body = self.call("GET", "/v1/tables/users/rows/99")
        self.assertEqual(status, 404)
        self.assertIn("not found", body["error"])
        self.assertEqual(self.call("GET", "/v1/tables/nope/rows/1")[0], 404)

        status, body = self.call("POST", "/v1/query", {
            "table": "users", "where": [["age", ">=", 25]], "columns": ["id", "age"],
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["access"], "index")
        self.assertEqual(body["rows"], [{"id": 1, "age": 30}, {"id": 3, "age": 40}])

        status, body = self.call("POST", "/v1/query", {"table": "users", "where": [["age", "=", 20]]})
        self.assertEqual((status, body["access"]), (200, "index"))
        self.assertEqual([r["id"] for r in body["rows"]], [2])

        status, body = self.call("POST", "/v1/query", {"table": "users", "where": [["name", "!=", "ann"]], "limit": 1})
        self.assertEqual((status, body["access"], body["rows"]), (200, "scan", [{"id": 2, "name": "bob", "age": 20}]))

        status, body = self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 1, "name": "dup", "age": 1}]})
        self.assertEqual(status, 409)
        self.assertIn("duplicate primary key", body["error"])
        status, body = self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 4, "name": "ann", "age": 1}]})
        self.assertEqual(status, 409)
        self.assertEqual(len(self.call("POST", "/v1/query", {"table": "users"})[1]["rows"]), 3)

    def test_transactions_conflict_and_rollback(self):
        self.create()
        self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 1, "name": "ann", "age": 30}]})
        old_txid = self.call("GET", "/v1/audit")[1]["entries"][-1]["txid"]

        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "users", "row": {"id": 2, "name": "bob", "age": 20}},
            {"op": "insert", "table": "users", "row": {"id": 3, "name": "cid", "age": 40}},
        ]})
        self.assertEqual(status, 200)
        self.assertTrue(body["committed"])
        self.assertGreater(body["lsn"], 0)

        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "update", "table": "users", "pk": 2, "patch": {"age": 21}},
            {"op": "delete", "table": "users", "pk": 1},
        ]})
        self.assertEqual(status, 200)
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/2")[1]["age"], 21)

        # a transaction pinned to an older snapshot must be rejected
        status, body = self.call("POST", "/v1/tx", {
            "snapshot_txid": old_txid,
            "ops": [{"op": "update", "table": "users", "pk": 2, "patch": {"age": 99}}],
        })
        self.assertEqual(status, 409)
        self.assertIn("after transaction", body["error"])
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/2")[1]["age"], 21)

        status, body = self.call("POST", "/v1/tx", {"ops": [
            {"op": "insert", "table": "users", "row": {"id": 5, "name": "eve", "age": 50}},
            {"op": "insert", "table": "users", "row": {"id": 3, "name": "dup", "age": 1}},
        ]})
        self.assertEqual(status, 409)
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/5")[0], 404)  # rolled back

        status, body = self.call("POST", "/v1/tx", {"ops": [{"op": "explode", "table": "users"}]})
        self.assertEqual(status, 400)

    def test_tx_isolation_parameter(self):
        self.create()
        self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 1, "name": "ann", "age": 30}]})
        old_txid = self.call("GET", "/v1/audit")[1]["entries"][-1]["txid"]
        op = {"op": "update", "table": "users", "pk": 1, "patch": {"age": 31}}

        status, body = self.call("POST", "/v1/tx", {"ops": [op], "isolation": "serializable"})
        self.assertEqual(status, 200)
        self.assertTrue(body["committed"])
        status, body = self.call("POST", "/v1/tx", {"ops": [op], "isolation": "snapshot"})
        self.assertEqual(status, 200)
        status, body = self.call("POST", "/v1/tx", {"ops": [op]})
        self.assertEqual(status, 200)

        for bad in ("serial", "SERIALIZABLE", 5, True, ["snapshot"]):
            status, body = self.call("POST", "/v1/tx", {"ops": [op], "isolation": bad})
            self.assertEqual(status, 400)
            self.assertIn("error", body)
        status, body = self.call("POST", "/v1/tx", {
            "ops": [op], "isolation": "serializable", "snapshot_txid": old_txid,
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, body = self.call("POST", "/v1/tx", {
            "ops": [op], "isolation": "serializable", "snapshot_txid": None,
        })
        self.assertEqual(status, 200)  # a null snapshot_txid is no pinning
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/1")[1]["age"], 31)

    def test_verify_audit_and_error_shapes(self):
        self.create()
        self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 1, "name": "ann", "age": 30}]})

        status, body = self.call("GET", "/v1/verify")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(body), ["crc_ok", "ok", "pages", "wal_records"])
        self.assertTrue(body["ok"])

        status, body = self.call("GET", "/v1/audit")
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], len(body["entries"]))
        self.assertEqual(body["entries"][-1]["ops"][0]["op"], "insert")
        self.assertEqual(body["entries"][-1]["at"], 7000)
        self.assertEqual(self.call("GET", "/v1/audit?limit=1")[1]["count"], 1)

        self.assertEqual(self.call("GET", "/v1/nope")[0], 404)
        status, body = self.call("POST", "/v1/query", {"table": "users", "where": [["bogus", "=", 1]]})
        self.assertEqual(status, 400)
        self.assertIn("unknown column", body["error"])
        status, body = self.call("POST", "/v1/tables/users/rows", {"rows": []})
        self.assertEqual(status, 400)
        self.assertIn("rows", body["error"])

        request = urllib.request.Request(
            "http://127.0.0.1:%d/v1/tables" % self.port, data=b"{not json", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                self.fail("expected an error status, got %d" % response.status)
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            self.assertIn("error", json.loads(exc.read().decode("utf-8")))

    def test_state_survives_a_restart(self):
        self.create()
        self.call("POST", "/v1/tables/users/rows", {"rows": [{"id": 1, "name": "ann", "age": 30}]})
        self.engine.reopen()
        self.assertEqual(self.call("GET", "/v1/tables/users/rows/1")[1]["name"], "ann")
        self.assertTrue(self.call("GET", "/v1/verify")[1]["ok"])


if __name__ == "__main__":
    unittest.main()
