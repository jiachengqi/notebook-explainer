"""One run: parse → static facts → prompts → generate (in parallel) → validate → repair → cover → render.

The model is the only non-deterministic step. Everything before it prepares
evidence, and everything after it checks the answer before a reader sees it.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import prompts
from .facts import Facts, redact_text
from .notebook import Notebook, reveal_hidden
from .providers import Completion, Provider, ProviderError
from .render import render_markdown
from .schema import DOC_SCHEMA, REPAIR_SCHEMA, ShapeError, check_claim as check_claim_shape
from .schema import check_document, extract_json
from .validate import check_claim, coverage_report, known_numbers, static_claim, uncovered_flags

FAILING = {"invalid", "unverified"}
CHARS_PER_TOKEN = 2.2   # measured on Gemini with this prompt (code with line labels is token-dense)


@dataclass
class Settings:
    parallel: bool = True
    cache: bool = True
    max_repair_rounds: int = 2
    on_unverified: str = "flag"            # "flag" keeps the claim with a warning; "drop" removes it
    coverage_audiences: tuple[str, ...] = ("agent",)
    max_prompt_tokens: int = 120_000
    local_only: bool = False               # refuse any provider that sends code off this machine
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path(".cache/llm")


class Cache:
    """Answers keyed by provider, model, settings and the exact prompt, so re-runs are instant and free."""

    def __init__(self, root: Path):
        self.root = root

    def key(self, provider: Provider, system: str, user: str, schema: dict | None) -> str:
        d = provider.describe()
        knobs = json.dumps({k: provider.cfg.get(k) for k in ("reasoning_effort", "response_format", "max_tokens")},
                           sort_keys=True)
        return prompts.prompt_hash(d["kind"], d["model"], d["base_url"], str(provider.temperature), knobs,
                                   system, user, json.dumps(schema, sort_keys=True))

    def get(self, key: str) -> dict | None:
        p = self.root / f"{key}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, comp: Completion) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps(comp.__dict__, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunState:
    notebook: Notebook
    facts: Facts
    provider: Provider
    audiences: list[str]
    source: str | None = None              # where the notebook was loaded from, for the report
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "queued"                 # queued | running | done | error
    events: list[dict] = field(default_factory=list)
    docs: dict[str, dict] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    error: str | None = None
    out_dir: str | None = None
    started: float = field(default_factory=time.time)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0})
    prompt_hashes: dict = field(default_factory=dict)
    flows: dict = field(default_factory=dict)   # host -> what was sent there
    notes: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, stage: str, msg: str, audience: str | None = None) -> None:
        with self._lock:
            self.events.append({"t": round(time.time() - self.started, 1), "stage": stage,
                                "audience": audience, "msg": msg})

    def public(self) -> dict:
        with self._lock:
            return {"id": self.id, "status": self.status, "events": list(self.events), "docs": self.docs,
                    "report": self.report, "error": self.error, "out_dir": self.out_dir,
                    "provider": self.provider.describe()}


# ---------------------------------------------------------------------------

def run(state: RunState, settings: Settings, export_dir: Path | None = None) -> RunState:
    state.status = "running"
    state.started = time.time()
    cache = Cache(settings.cache_dir)
    p = state.provider.describe()
    state.log("start", f"{state.notebook.name}: {len(state.notebook.cells)} cells; provider {p['name']} "
                       f"({p['model']}); {len(state.facts.flags)} static red flags")
    try:
        if settings.local_only and not state.provider.is_local:
            raise ProviderError(f"local_only is on, and provider '{p['name']}' sends the notebook to "
                                f"{state.provider.host}. Use ollama or offline, or turn local_only off.")
        workers = len(state.audiences) if settings.parallel else 1
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_generate, state, a, settings, cache): a for a in state.audiences}
            errors = []
            for fut in as_completed(futures):
                audience = futures[fut]
                try:
                    doc = fut.result()
                    with state._lock:
                        state.docs[audience] = doc
                    state.log("done", f"{doc['stats']['claims']} claims, {doc['stats']['verified']} verified",
                              audience)
                except (ProviderError, ShapeError) as exc:
                    errors.append(f"{audience}: {exc}")
                    state.log("error", str(exc), audience)
        if errors and not state.docs:
            raise ProviderError("; ".join(errors))
        state.report = _report(state, settings, errors)
        state.out_dir = str(_save(state, settings, export_dir))
        state.status = "done"
        state.log("saved", f"documents written to {state.out_dir}")
    except Exception as exc:  # the UI must always get an answer
        state.status = "error"
        state.error = str(exc)
        state.log("error", str(exc))
    return state


def _call(state: RunState, settings: Settings, cache: Cache, system: str, user: str, schema: dict,
          context: dict, label: str, audience: str) -> Completion:
    provider = state.provider
    use_cache = settings.cache and provider.kind != "offline"
    key = cache.key(provider, system, user, schema)
    with state._lock:
        state.prompt_hashes[f"{audience}:{label}"] = key
    if use_cache:
        hit = cache.get(key)
        if hit:
            with state._lock:
                state.usage["cache_hits"] += 1
            state.log(label, "cache hit (same provider, model and prompt)", audience)
            return Completion(**hit)
    state.log(label, f"calling {provider.name} ({provider.model})", audience)
    t0 = time.time()
    comp = provider.complete(system, user, schema, context)
    with state._lock:
        state.usage["calls"] += 1
        state.usage["input_tokens"] += comp.input_tokens
        state.usage["output_tokens"] += comp.output_tokens
        state.notes += [f"{audience}: {n}" for n in comp.notes]
        if comp.host:
            flow = state.flows.setdefault(comp.host, {"provider": provider.name, "model": provider.model,
                                                      "calls": 0, "request_bytes": 0, "input_tokens": 0,
                                                      "output_tokens": 0})
            flow["calls"] += 1
            flow["request_bytes"] += comp.request_bytes
            flow["input_tokens"] += comp.input_tokens
            flow["output_tokens"] += comp.output_tokens
    state.log(label, f"answer in {time.time() - t0:.1f}s ({comp.input_tokens} in / {comp.output_tokens} out tokens)"
              + (f"; {'; '.join(comp.notes)}" if comp.notes else ""), audience)
    if use_cache:
        cache.put(key, comp)
    return comp


def _generate(state: RunState, audience: str, settings: Settings, cache: Cache) -> dict:
    nb, facts = state.notebook, state.facts
    system = prompts.system_prompt()
    user = prompts.document_prompt(nb, facts, audience)
    context = {"audience": audience, "notebook": nb, "facts": facts}
    approx_tokens = int((len(system) + len(user) + len(json.dumps(DOC_SCHEMA))) / CHARS_PER_TOKEN)
    state.log("prompt", f"about {approx_tokens:,} input tokens (estimate)", audience)
    if approx_tokens > settings.max_prompt_tokens:
        raise ProviderError(f"The prompt is about {approx_tokens:,} tokens, above max_prompt_tokens "
                            f"({settings.max_prompt_tokens:,}). Raise the limit in harness.toml or use a model "
                            "with a larger context window.")

    doc = None
    for attempt in range(2):
        comp = _call(state, settings, cache, system, user, DOC_SCHEMA, context,
                     "generate" if attempt == 0 else "regenerate", audience)
        try:
            doc, _ = check_document(extract_json(comp.text))
            _scrub(doc)
            break
        except ShapeError as exc:
            state.log("validate", f"answer unusable: {exc}", audience)
            if attempt == 1:
                raise
            user = user + f"\n\n<note>Your previous answer could not be used: {exc}. Return the complete JSON " \
                          "object only.</note>"

    ignore = known_numbers(nb)
    prefix = audience[0]
    n = 0
    for section in doc["sections"]:
        for claim in section["claims"]:
            n += 1
            claim["id"] = f"{prefix}{n}"
            claim["origin"] = "model"
            _apply_check(nb, claim, ignore, facts)
    state.log("validate", _status_line(doc), audience)

    rounds = 0
    for rounds in range(1, settings.max_repair_rounds + 1):
        failed = [c for s in doc["sections"] for c in s["claims"] if c["status"] in FAILING]
        if not failed:
            rounds -= 1
            break
        state.log("repair", f"round {rounds}: {len(failed)} claims sent back", audience)
        try:
            comp = _call(state, settings, cache, system, prompts.repair_prompt(nb, facts, failed), REPAIR_SCHEMA,
                         {**context, "repair": True, "failed": failed}, f"repair {rounds}", audience)
            fixes = extract_json(comp.text).get("claims", [])
        except (ProviderError, ShapeError) as exc:
            state.log("repair", f"repair failed: {exc}", audience)
            break
        by_id = {c["id"]: c for c in failed}
        for fix in fixes if isinstance(fixes, list) else []:
            if not isinstance(fix, dict) or fix.get("id") not in by_id:
                continue
            claim = by_id[fix["id"]]
            claim["repair_rounds"] = claim.get("repair_rounds", 0) + 1
            if fix.get("drop"):
                claim["status"] = "dropped"
                claim["problems"] = claim.get("problems", []) + ["dropped by the model during repair"]
                continue
            fixed, problems = check_claim_shape(fix, fix["id"])
            if fixed is None:
                continue
            _scrub_claim(fixed)
            for key in ("text", "kind", "citations", "evidence"):
                claim[key] = fixed[key]
            claim["shape_problems"] = problems
            claim["repaired"] = True
            _apply_check(nb, claim, ignore, facts)
        state.log("validate", _status_line(doc), audience)

    for claim in (c for s in doc["sections"] for c in s["claims"]):
        if claim["status"] in FAILING and settings.on_unverified == "drop":
            claim["status"] = "dropped"

    if audience in settings.coverage_audiences:
        missing = uncovered_flags(doc, facts.flags_by_severity("high", "medium"))
        if missing:
            added = []
            for flag in missing:
                claim = static_claim(nb, flag)
                n += 1
                claim["id"] = f"{prefix}{n}"
                _apply_check(nb, claim, ignore, facts)
                added.append(claim)
            doc["sections"].append({"heading": "Also detected by static analysis", "claims": added})
            state.log("coverage", f"added {len(added)} red flags the model did not cite", audience)

    doc["audience"] = audience
    doc["repair_rounds"] = max(rounds, 0)
    doc["coverage"] = coverage_report(doc, facts)
    doc["stats"] = _stats(doc)
    return doc


def _scrub(doc: dict) -> None:
    """Model text is untrusted too: secrets and invisible characters never pass through it."""
    doc["title"] = reveal_hidden(redact_text(doc.get("title", "")))
    for section in doc["sections"]:
        section["heading"] = reveal_hidden(redact_text(section.get("heading", "")))
        for claim in section["claims"]:
            _scrub_claim(claim)


def _scrub_claim(claim: dict) -> None:
    for key in ("text", "evidence"):
        if isinstance(claim.get(key), str):
            claim[key] = reveal_hidden(redact_text(claim[key]))


def _apply_check(nb: Notebook, claim: dict, ignore: set[float], facts: Facts | None = None) -> None:
    result = check_claim(nb, claim, ignore, facts)
    claim["status"] = result.status
    claim["problems"] = result.problems
    claim["evidence_found"] = result.evidence_found


def _stats(doc: dict) -> dict:
    claims = [c for s in doc["sections"] for c in s["claims"]]
    count = lambda st: sum(1 for c in claims if c["status"] == st)  # noqa: E731
    shown = [c for c in claims if c["status"] != "dropped"]
    return {
        "claims": len(shown), "verified": count("verified"), "partial": count("partial"),
        "unchecked": count("unchecked"), "unverified": count("unverified"), "invalid": count("invalid"),
        "dropped": count("dropped"), "repaired": sum(1 for c in claims if c.get("repaired")),
        "static_added": sum(1 for c in claims if c.get("origin") == "static"),
        "citations": sum(len(c["citations"]) for c in shown),
    }


def _status_line(doc: dict) -> str:
    s = _stats(doc)
    return (f"{s['claims']} claims: {s['verified']} verified, {s['partial']} partial, {s['unchecked']} unchecked, "
            f"{s['unverified']} unverified, {s['invalid']} invalid citations")


def _report(state: RunState, settings: Settings, errors: list[str]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - state.started, 1),
        "notebook": {**state.notebook.summary(), "path": state.source},
        "provider": state.provider.describe(),
        "usage": state.usage,
        "data_sent": state.flows,
        "settings": {"max_repair_rounds": settings.max_repair_rounds, "on_unverified": settings.on_unverified,
                     "cache": settings.cache, "parallel": settings.parallel, "local_only": settings.local_only},
        "documents": {a: {**d["stats"], "repair_rounds": d["repair_rounds"], "coverage": d["coverage"]}
                      for a, d in state.docs.items()},
        "static_flags": [{"id": f.id, "severity": f.severity, "title": f.title, "citations": f.citations}
                         for f in state.facts.flags],
        "redacted_lines": [f"C{c}:L{l}" for (c, l) in state.facts.redactions],
        "injection_lines": sorted(f"C{c}:L{l}" for (c, l) in state.facts.injection_lines()),
        "hidden_unicode_lines": [f"C{h['cell']}:L{h['line']}" for h in state.notebook.hidden],
        "prompt_hashes": state.prompt_hashes,
        "notes": state.notes,
        "errors": errors,
    }


def _save(state: RunState, settings: Settings, export_dir: Path | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = Path(state.notebook.name).stem
    out = settings.runs_dir / f"{stem}-{state.notebook.sha256[:8]}-{state.provider.name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    for audience, doc in state.docs.items():
        (out / f"{audience}.md").write_text(render_markdown(state.notebook, doc, state.report), encoding="utf-8")
    (out / "docs.json").write_text(json.dumps(state.docs, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(state.report, indent=1, ensure_ascii=False), encoding="utf-8")
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("eval.json", "verify.json"):     # they described the previous run
            (export_dir / stale).unlink(missing_ok=True)
        for f in out.iterdir():
            shutil.copy2(f, export_dir / f.name)
    return out
