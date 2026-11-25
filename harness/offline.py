"""Documents written from static analysis only, without a model.

Used by the ``offline`` provider: by the test suite and CI (no keys needed), and
as a fallback during a demo if every model provider is unavailable. Every claim
here is built from a fact that carries its own citation, so the documents are
correct but shallow. They describe what the code touches, not what it means.
"""

from __future__ import annotations

from .facts import SEVERITY_ORDER, Facts
from .notebook import Notebook, format_citation, parse_citation


def _evidence(nb: Notebook, cite: str) -> str:
    parsed = parse_citation(cite)
    if not parsed:
        return ""
    return nb.line_text(parsed[0], parsed[1]).strip()[:100]


def _claim(nb: Notebook, text: str, citations: list[str], kind: str = "fact") -> dict:
    citations = [c for c in citations if c][:8]
    return {"text": text, "kind": kind, "citations": citations, "evidence": _evidence(nb, citations[0])}


def offline_answer(ctx: dict) -> dict:
    if ctx.get("repair"):
        # Offline claims are built from cited facts; anything that still fails is dropped.
        return {"claims": [{**c, "drop": True} for c in ctx.get("failed", [])]}
    nb: Notebook = ctx["notebook"]
    facts: Facts = ctx["facts"]
    audience = ctx["audience"]
    builders = {"manager": _manager, "developer": _developer, "agent": _agent}
    sections = [s for s in builders[audience](nb, facts) if s["claims"]]
    titles = {
        "manager": "What this notebook does and what could go wrong",
        "developer": "How this notebook is built",
        "agent": "Working safely on this notebook",
    }
    return {"title": f"{titles[audience]} (static analysis only, no LLM)", "sections": sections}


def _touches(nb: Notebook, facts: Facts) -> list[dict]:
    claims = []
    for w in facts.widgets:
        claims.append(_claim(nb, f"It takes a run option `{w['name']}` with default {w['default']}.", [w["cite"]]))
    for r in facts.runs:
        claims.append(_claim(nb, f"It loads shared code with `%run {r['target']}`, which is not visible here.",
                             [r["cite"]], "external"))
    for r in facts.reads:
        claims.append(_claim(nb, f"It reads {r['what']}.", [r["cite"]]))
    for w in facts.writes:
        guard = f" only when `{w['guard']}`" if w.get("guard") else " on every run"
        claims.append(_claim(nb, f"It writes table `{w['target']}` with mode {w['mode']}{guard}.",
                             [w["cite"]] + ([w["target_cite"]] if w.get("target_cite") else [])))
    return claims


def _risks(nb: Notebook, facts: Facts, levels: tuple[str, ...]) -> list[dict]:
    flags = sorted(facts.flags_by_severity(*levels), key=lambda f: SEVERITY_ORDER[f.severity])
    return [_claim(nb, f.detail, f.citations, "risk") for f in flags]


def _manager(nb: Notebook, facts: Facts) -> list[dict]:
    questions = [_claim(nb, f.question, f.citations, "question")
                 for f in facts.flags_by_severity("high", "medium") if f.question]
    return [
        {"heading": "What the notebook touches", "claims": _touches(nb, facts)},
        {"heading": "What could go wrong (found by static analysis)", "claims": _risks(nb, facts, ("high", "medium"))},
        {"heading": "Questions for the owner", "claims": questions},
    ]


def _developer(nb: Notebook, facts: Facts) -> list[dict]:
    structure = []
    defs_by_cell: dict[int, list[str]] = {}
    for name, f in facts.functions.items():
        cell = parse_citation(f["cite"])[0]
        defs_by_cell.setdefault(cell, []).append(name)
    for cell_idx, names in sorted(defs_by_cell.items()):
        first = facts.functions[names[0]]["cite"]
        cites = [facts.functions[n]["cite"] for n in names]
        structure.append(_claim(nb, f"Cell {cell_idx} defines {', '.join(f'`{n}`' for n in names)}.",
                                [first] + cites[1:]))
    for name, items in facts.constants.items():
        structure.append(_claim(nb, f"`{name}` is set to `{items[0]['value']}`.", [items[0]["cite"]]))
    for cell in nb.cells:
        if cell.lang == "md" and len(cell.lines) > 1:
            structure.append(_claim(nb, f"Cell {cell.idx} is documentation: \"{cell.lines[1].text.strip()[:60]}\".",
                                    [format_citation(cell.idx, 2)]))
    flow = [_claim(nb, f"`{x['name']}` is built at {', '.join(x['defined'][:3])} and used later in cells "
                       f"{', '.join(str(c) for c in x['used_in_cells'][:8])}.", x["defined"][:3])
            for x in facts.lineage]
    return [
        {"heading": "Inputs and outputs", "claims": _touches(nb, facts)},
        {"heading": "Structure", "claims": structure},
        {"heading": "Data flow between cells", "claims": flow},
        {"heading": "Gotchas", "claims": _risks(nb, facts, ("medium", "low"))},
    ]


def _agent(nb: Notebook, facts: Facts) -> list[dict]:
    checks = []
    for f in facts.flags:
        if f.kind == "print-only-check":
            checks.append(_claim(nb, f"Existing check: {f.title.split(' only prints')[0]} prints a message when it "
                                     "fails; after a change, its output must not get worse.", f.citations[:1]))
        if f.kind == "constant-exit":
            checks.append(_claim(nb, "The notebook ends with `dbutils.notebook.exit`; compare its output before "
                                     "and after a change.", f.citations[:1]))
    for w in facts.writes:
        if w.get("guard"):
            checks.append(_claim(nb, f"Table writes only run when `{w['guard']}`, so test changes with that "
                                     "condition false.", [w["cite"]]))
            break
    settings = [_claim(nb, f"Do not change `{name}` (set to `{items[-1]['value']}`) without the owner's approval.",
                       [i["cite"] for i in items], "fact") for name, items in facts.constants.items()]
    return [
        {"heading": "Risky areas", "claims": _risks(nb, facts, ("high", "medium"))},
        {"heading": "How to check a change is safe", "claims": checks},
        {"heading": "Do not change without approval", "claims": settings},
    ]
