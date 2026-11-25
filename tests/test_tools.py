"""verify, the red-team suite, the MCP server and the README results table."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness.app import main
from harness.evaluate import END, START, update_readme
from harness.facts import analyze
from harness.notebook import load_path
from harness.pipeline import RunState, Settings, run
from harness.providers import make_provider
from harness.redteam import run_all
from harness.verify import verify

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "cash_application_engine.py"


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        nb = load_path(SAMPLE)
        settings = Settings(runs_dir=Path(cls.tmp.name) / "runs", cache_dir=Path(cls.tmp.name) / "cache")
        state = run(RunState(nb, analyze(nb), make_provider("offline", {"kind": "offline"}), ["agent"]), settings)
        cls.run_dir = Path(state.out_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unchanged_notebook_has_no_stale_claims(self):
        result = verify(self.run_dir, SAMPLE)
        self.assertFalse(result["notebook_changed"])
        self.assertEqual(result["worse"], 0)

    def test_edited_notebook_shows_stale_claims_and_fails(self):
        edited = Path(self.tmp.name) / "edited.py"
        edited.write_text(SAMPLE.read_text(encoding="utf-8").replace("TOL = 5.0", "TOL_NEW = 2.0"), encoding="utf-8")
        result = verify(self.run_dir, edited)
        self.assertTrue(result["notebook_changed"])
        self.assertGreater(result["worse"], 0)
        self.assertTrue(any("C29:L4" in c["citations"] for c in result["changes"]))
        out = Path(self.tmp.name) / "rechecked"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["verify", str(self.run_dir), "--notebook", str(edited), "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertTrue((out / "agent.md").is_file() and (out / "verify.json").is_file())


class RedTeamTests(unittest.TestCase):
    def test_every_attack_is_contained(self):
        results = run_all()
        self.assertGreaterEqual(len(results), 10)
        failed = [(r["id"], r["problems"]) for r in results if not r["passed"]]
        self.assertEqual(failed, [])


class MCPTests(unittest.TestCase):
    def rpc(self, *messages):
        lines = "".join(json.dumps(m) + "\n" for m in messages)
        proc = subprocess.run([sys.executable, "-m", "harness", "mcp", "--root", str(ROOT)], input=lines,
                              capture_output=True, text=True, cwd=ROOT, timeout=60)
        return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

    def test_handshake_list_and_call(self):
        replies = self.rpc(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "red_flags", "arguments": {"path": "samples/cash_application_engine.py"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "explain_cell", "arguments": {"path": "samples/cash_application_engine.py",
                                                              "cell": 29}}},
        )
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4])   # no reply to the notification
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("check_change", {t["name"] for t in replies[1]["result"]["tools"]})
        flags = json.loads(replies[2]["result"]["content"][0]["text"])["flags"]
        self.assertTrue(any("TOL" in f["title"] for f in flags))
        cell = json.loads(replies[3]["result"]["content"][0]["text"])
        self.assertEqual(cell["lines"][3], {"address": "C29:L4", "file_line": 361, "text": "TOL = 5.0"})

    def test_paths_outside_the_root_are_refused(self):
        replies = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "red_flags", "arguments": {"path": "../../etc/passwd"}}})
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertIn("outside", replies[0]["result"]["content"][0]["text"])

    def test_check_change_lists_touched_flags(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            after = Path(tmp) / "after.py"
            after.write_text(SAMPLE.read_text(encoding="utf-8").replace("TOL = 5.0", "TOL = 2.0"), encoding="utf-8")
            replies = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": "check_change",
                                           "arguments": {"before": "samples/cash_application_engine.py",
                                                         "after": str(after.relative_to(ROOT))}}})
        result = json.loads(replies[0]["result"]["content"][0]["text"])
        self.assertEqual(result["changed"], ["C29:L4"])
        self.assertTrue(any("TOL" in f["title"] for f in result["red_flags_touched"]))


class ReadmeTableTests(unittest.TestCase):
    def test_results_block_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(f"intro\n{START}\nold\n{END}\nend\n", encoding="utf-8")
            self.assertTrue(update_readme(readme, "| new |"))
            self.assertEqual(readme.read_text(encoding="utf-8"), f"intro\n{START}\n| new |\n{END}\nend\n")


if __name__ == "__main__":
    unittest.main()
