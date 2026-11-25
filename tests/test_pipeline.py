"""End-to-end runs: offline provider, and a scripted model that needs a repair."""

import json
import tempfile
import unittest
from pathlib import Path

from harness.facts import analyze
from harness.notebook import load_path
from harness.providers import Completion, Provider, make_provider
from harness.pipeline import RunState, Settings, run
from harness.schema import ShapeError, check_document, extract_json
from harness.validate import check_claim

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "cash_application_engine.py"


class ScriptedProvider(Provider):
    """Plays back answers in order: lets tests drive the repair loop without a real model."""
    kind = "scripted"

    def __init__(self, answers):
        super().__init__("scripted", {"model": "script"})
        self.answers = list(answers)
        self.prompts = []

    def complete(self, system, user, schema=None, context=None):
        self.prompts.append(user)
        return Completion(json.dumps(self.answers.pop(0)), 100, 50, "script")


def doc(*claims):
    return {"title": "Test", "sections": [{"heading": "Rules", "claims": list(claims)}]}


class SchemaTests(unittest.TestCase):
    def test_extract_json_tolerates_fences(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Here you go: {"a": 1} hope it helps'), {"a": 1})
        with self.assertRaises(ShapeError):
            extract_json("no json here")

    def test_claim_without_citation_is_flagged(self):
        d, problems = check_document(doc({"text": "x", "kind": "fact", "citations": [], "evidence": ""}))
        self.assertTrue(any("at least one address" in p for p in problems))


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = load_path(SAMPLE)

    def check(self, text, cites, evidence):
        return check_claim(self.nb, {"text": text, "citations": cites, "evidence": evidence})

    def test_verified(self):
        r = self.check("`TOL` is reassigned from 0.5 to 5.0.", ["C4:L1", "C29:L4"], "TOL = 5.0")
        self.assertEqual(r.status, "verified")

    def test_out_of_range(self):
        r = self.check("Something.", ["C26:L80"], "x")
        self.assertEqual(r.status, "invalid")
        self.assertIn("cell 26 has 56 lines", r.problems[0])

    def test_invented_evidence(self):
        r = self.check("It emails the AR team.", ["C40:L1"], "send_email(team)")
        self.assertEqual(r.status, "unverified")

    def test_names_must_be_in_cited_lines(self):
        r = self.check("Matching uses `MAX_WO`.", ["C4:L5"], "EXCL")
        self.assertEqual(r.status, "unverified")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(runs_dir=Path(self.tmp.name) / "runs", cache_dir=Path(self.tmp.name) / "cache")
        self.nb = load_path(SAMPLE)
        self.facts = analyze(self.nb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_offline_run_produces_three_cited_documents(self):
        state = RunState(self.nb, self.facts, make_provider("offline", {"kind": "offline"}),
                         ["manager", "developer", "agent"])
        run(state, self.settings, export_dir=Path(self.tmp.name) / "export")
        self.assertEqual(state.status, "done", state.error)
        for audience in ("manager", "developer", "agent"):
            d = state.docs[audience]
            self.assertGreater(d["stats"]["claims"], 10)
            self.assertEqual(d["stats"]["invalid"], 0)
            self.assertEqual(d["stats"]["unverified"], 0)
            md = (Path(state.out_dir) / f"{audience}.md").read_text()
            self.assertIn("## Code references", md)
        self.assertEqual(state.docs["agent"]["coverage"]["missing"], [])
        self.assertTrue((Path(self.tmp.name) / "export" / "report.json").is_file())

    def test_repair_fixes_bad_citation_and_coverage_adds_missed_flags(self):
        good = {"text": "`TOL` is reassigned from 0.5 to 5.0.", "kind": "risk",
                "citations": ["C4:L1", "C29:L4"], "evidence": "TOL = 5.0"}
        bad = {"text": "`MAX_WO` caps write-offs at 250.", "kind": "fact",
               "citations": ["C4:L9"], "evidence": "MAX_WO = 250.0"}
        fixed = {"id": "a2", "drop": False, "text": "`MAX_WO` is set to 250 but never used.", "kind": "risk",
                 "citations": ["C4:L2"], "evidence": "MAX_WO = 250.0"}
        provider = ScriptedProvider([doc(good, bad), {"claims": [fixed]}])
        state = RunState(self.nb, self.facts, provider, ["agent"])
        run(state, self.settings)
        self.assertEqual(state.status, "done", state.error)
        agent = state.docs["agent"]
        claims = {c["id"]: c for s in agent["sections"] for c in s["claims"]}
        self.assertEqual(claims["a2"]["status"], "verified")
        self.assertTrue(claims["a2"]["repaired"])
        self.assertIn("C4:L9", provider.prompts[1])                  # the repair prompt names the problem
        self.assertIn("cell 4 has 8 lines", provider.prompts[1])
        self.assertEqual(agent["repair_rounds"], 1)
        self.assertGreater(agent["stats"]["static_added"], 5)       # missed red flags were added
        self.assertEqual(agent["coverage"]["missing"], [])

    def test_unrepairable_claim_is_flagged_not_hidden(self):
        bad = {"text": "It sends an email to the CFO.", "kind": "fact", "citations": ["C4:L1"],
               "evidence": "send_email(cfo)"}
        provider = ScriptedProvider([doc(bad), {"claims": []}, {"claims": []}])
        state = RunState(self.nb, self.facts, provider, ["manager"])
        run(state, self.settings)
        claim = state.docs["manager"]["sections"][0]["claims"][0]
        self.assertEqual(claim["status"], "unverified")
        self.assertEqual(state.docs["manager"]["stats"]["unverified"], 1)

    def test_cache_hit_on_rerun(self):
        good = {"text": "`TOL` is reassigned from 0.5 to 5.0.", "kind": "risk",
                "citations": ["C4:L1", "C29:L4"], "evidence": "TOL = 5.0"}
        first = ScriptedProvider([doc(good)])
        first.kind = "openai_compat"
        run(RunState(self.nb, self.facts, first, ["developer"]), self.settings)
        second = ScriptedProvider([])                               # would fail if called
        second.kind = "openai_compat"
        state = RunState(self.nb, self.facts, second, ["developer"])
        run(state, self.settings)
        self.assertEqual(state.status, "done", state.error)
        self.assertEqual(state.usage["cache_hits"], 1)

    def test_cache_misses_when_reasoning_effort_changes(self):
        good = {"text": "`TOL` is reassigned from 0.5 to 5.0.", "kind": "risk",
                "citations": ["C4:L1", "C29:L4"], "evidence": "TOL = 5.0"}
        low = ScriptedProvider([doc(good)])
        low.kind, low.cfg["reasoning_effort"] = "openai_compat", "low"
        run(RunState(self.nb, self.facts, low, ["developer"]), self.settings)
        high = ScriptedProvider([doc(good)])
        high.kind, high.cfg["reasoning_effort"] = "openai_compat", "high"
        state = RunState(self.nb, self.facts, high, ["developer"])
        run(state, self.settings)
        self.assertEqual(state.status, "done", state.error)
        self.assertEqual(state.usage["cache_hits"], 0)    # a stale "low" answer must not come back
        self.assertEqual(state.usage["calls"], 1)


if __name__ == "__main__":
    unittest.main()


class PolicyAndLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.nb = load_path(SAMPLE)
        self.facts = analyze(self.nb)

    def tearDown(self):
        self.tmp.cleanup()

    def settings(self, **kw):
        return Settings(runs_dir=Path(self.tmp.name) / "runs", cache_dir=Path(self.tmp.name) / "cache", **kw)

    def test_local_only_refuses_a_cloud_provider(self):
        gemini = make_provider("gemini", {"kind": "openai_compat", "api_key_env": "",
                                          "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"})
        state = run(RunState(self.nb, self.facts, gemini, ["manager"]), self.settings(local_only=True))
        self.assertEqual(state.status, "error")
        self.assertIn("local_only", state.error)

    def test_prompt_above_the_cap_is_refused(self):
        state = run(RunState(self.nb, self.facts, make_provider("offline", {"kind": "offline"}), ["manager"]),
                    self.settings(max_prompt_tokens=1000))
        self.assertEqual(state.status, "error")
        self.assertIn("max_prompt_tokens", state.error)

    def test_report_records_what_was_sent_where(self):
        good = {"text": "`TOL` is reassigned from 0.5 to 5.0.", "kind": "risk",
                "citations": ["C4:L1", "C29:L4"], "evidence": "TOL = 5.0"}

        class Remote(ScriptedProvider):
            def complete(self, system, user, schema=None, context=None):
                comp = super().complete(system, user, schema, context)
                comp.host, comp.request_bytes = "api.example.test", 1234
                return comp

        state = run(RunState(self.nb, self.facts, Remote([doc(good)]), ["developer"]), self.settings(cache=False))
        self.assertEqual(state.report["data_sent"]["api.example.test"]["calls"], 1)
        self.assertEqual(state.report["data_sent"]["api.example.test"]["request_bytes"], 1234)


class RealisticAnswerTests(unittest.TestCase):
    """A hand-written answer in the style a model returns (tests/fixtures/model_answer_manager.json)."""

    def test_realistic_answer_with_repair(self):
        answer = json.loads((ROOT / "tests" / "fixtures" / "model_answer_manager.json").read_text())
        repair = {"claims": [
            {"id": "m7", "drop": False, "text": "If it names no invoice, it is matched to one open invoice with the "
             "same amount, within 0.01 USD.", "kind": "fact", "citations": ["C26:L20", "C4:L3"],
             "evidence": "EX_TOL = 0.01"},
            {"id": "m16", "drop": True, "text": "", "kind": "fact", "citations": ["C4:L2"], "evidence": ""},
        ]}
        tmp = tempfile.TemporaryDirectory()
        settings = Settings(runs_dir=Path(tmp.name) / "runs", cache_dir=Path(tmp.name) / "cache")
        nb = load_path(SAMPLE)
        state = RunState(nb, analyze(nb), ScriptedProvider([answer, repair]), ["manager"])
        run(state, settings)
        claims = {c["id"]: c for s in state.docs["manager"]["sections"] for c in s["claims"]}
        self.assertEqual(claims["m7"]["status"], "verified")         # missing citation added during repair
        self.assertEqual(claims["m16"]["status"], "dropped")         # invented approval rule
        # This claim is wrong in meaning: C17 computes a value-date rate, but C18:L2, which would use it, is
        # commented out. It names nothing checkable, so it is "unchecked", not verified. Its quote is real.
        self.assertEqual(claims["m5"]["status"], "unchecked")
        stats = state.docs["manager"]["stats"]
        self.assertEqual(stats["unverified"] + stats["invalid"], 0)
        tmp.cleanup()
