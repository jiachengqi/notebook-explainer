"""The local web server: routes, security guards, and a full offline run through the API."""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile

from harness.app import App, make_handler
from harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        config = load_config(str(ROOT / "harness.toml"))
        config.paths = {"runs_dir": str(Path(cls.tmp.name) / "runs"), "cache_dir": str(Path(cls.tmp.name) / "c")}
        app = App(config)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, 0))
        port = cls.server.server_address[1]
        cls.server.RequestHandlerClass = make_handler(app, port)
        cls.base = f"http://127.0.0.1:{port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers

    def post(self, path, body, header=True):
        headers = {"Content-Type": "application/json"}
        if header:
            headers["X-Harness"] = "1"
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_and_security_headers(self):
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Notebook Explainer", body)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])

    def test_rejects_foreign_host_and_missing_header(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/config", {"Host": "evil.example"})
        self.assertEqual(ctx.exception.code, 403)
        status, _ = self.post("/api/load", {"path": "x"}, header=False)
        self.assertEqual(status, 403)

    def test_static_path_traversal(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/static/../app.py")
        self.assertEqual(ctx.exception.code, 404)

    def test_bad_file_type(self):
        status, body = self.post("/api/load", {"path": str(ROOT / "harness.toml")})
        self.assertEqual(status, 400)
        self.assertIn("Unsupported file type", body["error"])

    def test_full_offline_run(self):
        status, loaded = self.post("/api/load", {"path": str(ROOT / "samples" / "cash_application_engine.py")})
        self.assertEqual(status, 200)
        self.assertEqual(loaded["notebook"]["cells"], 47)
        status, started = self.post("/api/generate", {"notebook_id": loaded["notebook_id"], "provider": "offline"})
        self.assertEqual(status, 200)
        for _ in range(100):
            _, body, _ = self.get(f"/api/runs/{started['run_id']}")
            state = json.loads(body)
            if state["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        self.assertEqual(state["status"], "done", state.get("error"))
        self.assertEqual(set(state["docs"]), {"manager", "developer", "agent"})
        status, md, _ = self.get(f"/api/runs/{started['run_id']}/agent.md")
        self.assertIn(b"## Code references", md)

    def test_upload_content(self):
        text = (ROOT / "samples" / "sql_example.sql").read_text()
        status, loaded = self.post("/api/load", {"filename": "sql_example.sql", "content": text})
        self.assertEqual(status, 200)
        self.assertEqual(loaded["notebook"]["format"], "databricks-sql")


if __name__ == "__main__":
    unittest.main()
