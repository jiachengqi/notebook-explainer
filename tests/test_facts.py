import unittest
from pathlib import Path

from harness.facts import analyze
from harness.notebook import load_path

ROOT = Path(__file__).resolve().parents[1]


def flag(facts, flag_id):
    return next((f for f in facts.flags if f.id == flag_id), None)


class CashAppFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = load_path(ROOT / "samples" / "cash_application_engine.py")
        cls.facts = analyze(cls.nb)

    def test_no_parse_errors(self):
        self.assertEqual(self.facts.parse_errors, [])

    def test_reassigned_constant(self):
        f = flag(self.facts, "reassigned-constant:TOL")
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "high")
        self.assertEqual(f.citations, ["C4:L1", "C29:L4"])
        self.assertIn("Functions defined earlier", f.detail)   # do_wo reads TOL at call time

    def test_unused_config_and_dead_code(self):
        self.assertIsNotNone(flag(self.facts, "unused-constant:MAX_WO"))
        self.assertIsNotNone(flag(self.facts, "unused-function:alloc_by_aging"))
        self.assertIsNone(flag(self.facts, "unused-function:nrm"))   # used through F.udf
        ids = {f.id for f in self.facts.flags}
        self.assertIn("commented-code:C15:L2-18", ids)
        self.assertIn("commented-code:C20:L1-21", ids)
        self.assertIn("commented-code:C18:L2", ids)

    def test_externals_come_from_run(self):
        self.assertEqual(set(self.facts.externals), {"get_gl_map", "notify_ar", "log_run"})
        self.assertEqual(self.facts.runs[0]["target"], "../shared/fin_utils")
        self.assertNotIn("StringType", self.facts.externals)   # from pyspark.sql.types import *

    def test_writes_modes_and_guard(self):
        writes = {w["target"]: w for w in self.facts.writes}
        self.assertEqual(writes["fin_cash.alloc_lines"]["mode"], "overwrite")
        self.assertTrue(writes["fin_cash.alloc_lines"]["overwrite_schema"])
        self.assertEqual(writes["fin_cash.alloc_lines"]["cite"], "C37:L2")
        self.assertEqual(writes["fin_cash.alloc_lines"]["guard"], 'dry == "N"')
        self.assertEqual(writes["fin_cash.ar_aging_snap"]["mode"], "append")

    def test_dry_run_side_effects(self):
        f = flag(self.facts, "ungated-call:notify_ar:C43:L11")
        self.assertIsNotNone(f)
        self.assertEqual(f.severity, "high")
        self.assertIsNone(flag(self.facts, "ungated-call:get_gl_map:C33:L3"))   # a lookup, not a side effect

    def test_checks_and_exit(self):
        self.assertEqual(flag(self.facts, "print-check:C41:L7-8").severity, "high")   # after the write
        self.assertEqual(flag(self.facts, "print-check:C19:L9-10").severity, "medium")
        self.assertIsNotNone(flag(self.facts, "constant-exit:C47:L1"))

    def test_state_and_scale(self):
        g = flag(self.facts, "global-state:bal")
        self.assertIn("C26:L12", g.citations)
        self.assertIsNotNone(flag(self.facts, "driver-side"))
        self.assertEqual(flag(self.facts, "nondeterministic:C13:L6").severity, "medium")
        self.assertEqual(flag(self.facts, "nondeterministic:C11:L2").severity, "low")

    def test_widgets(self):
        names = [w["name"] for w in self.facts.widgets]
        self.assertEqual(names, ["src", "run_dt", "co", "dry"])

    def test_fact_sheet_is_cited(self):
        sheet = self.facts.fact_sheet()
        self.assertIn("TOL: 0.5 at C4:L1; 5.0 at C29:L4", sheet)
        self.assertIn("%run ../shared/fin_utils at C5:L1", sheet)
        self.assertIn("nrm defined at C10:L1-9, passed as a value", sheet)


class SafetyFactsTests(unittest.TestCase):
    def test_injection_secret_and_misleading_notebook(self):
        nb = load_path(ROOT / "samples" / "injection_example.py")
        facts = analyze(nb)
        kinds = {f.kind for f in facts.flags}
        self.assertIn("prompt-injection", kinds)
        self.assertIn("secret", kinds)
        self.assertEqual(nb.line_text(2, 2), 'api_token = "«REDACTED»"')   # redacted in the notebook itself
        self.assertNotIn("dapi0123", nb.listing())
        self.assertIn("send_payment_file", facts.externals)

    def test_sql_notebook_tables(self):
        facts = analyze(load_path(ROOT / "samples" / "sql_example.sql"))
        targets = {w["target"] for w in facts.writes}
        self.assertEqual(targets, {"finance.daily_revenue", "finance.revenue_summary"})


if __name__ == "__main__":
    unittest.main()
