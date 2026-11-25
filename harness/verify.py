"""Re-check a saved run against a notebook.

    python -m harness verify RUN_DIR [--notebook FILE] [--out DIR]

Every claim is checked again with the current validator, with no model call.
Two uses:

* the notebook changed since the run: claims whose cited lines no longer support
  them are reported as stale, and the exit code is 1, so CI can block the change;
* the validator changed: old documents are re-scored with the new rules.

With --out, the documents are rendered again with the new statuses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .facts import analyze
from .notebook import NotebookError, load_path
from .pipeline import _stats
from .render import render_markdown
from .validate import check_claim, coverage_report, known_numbers

RANK = {"verified": 0, "partial": 1, "unchecked": 2, "unverified": 3, "invalid": 4}


def verify(run_dir: Path, notebook_path: Path | None = None) -> dict:
    docs_file = run_dir / "docs.json"
    if not docs_file.is_file():
        raise NotebookError(f"{run_dir} has no docs.json. Point at a run folder or docs/generated.")
    docs = json.loads(docs_file.read_text(encoding="utf-8"))
    report_file = run_dir / "report.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
    path = notebook_path or Path((report.get("notebook") or {}).get("path") or "")
    if not str(path) or not path.is_file():
        raise NotebookError("Could not find the notebook for this run. Give it with --notebook FILE.")
    nb = load_path(path)
    facts = analyze(nb)
    ignore = known_numbers(nb)
    before_sha = (report.get("notebook") or {}).get("sha256")
    changes, summary = [], {}
    for audience, doc in docs.items():
        before = dict(doc.get("stats") or _stats(doc))
        for section in doc["sections"]:
            for claim in section["claims"]:
                if claim.get("status") == "dropped":
                    continue
                old = claim.get("status")
                result = check_claim(nb, claim, ignore, facts)
                claim.update(status=result.status, problems=result.problems, evidence_found=result.evidence_found)
                if old != result.status:
                    changes.append({"document": audience, "id": claim.get("id"), "before": old,
                                    "after": result.status, "text": claim["text"], "citations": claim["citations"],
                                    "problems": result.problems})
        doc["stats"] = _stats(doc)
        doc["coverage"] = coverage_report(doc, facts)
        summary[audience] = {"before": before, "after": doc["stats"]}
    worse = [c for c in changes if RANK.get(c["after"], 9) > RANK.get(c["before"], 9)]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notebook": {**nb.summary(), "path": str(notebook_path or path)},
        "notebook_changed": bool(before_sha and before_sha != nb.sha256),
        "documents": summary,
        "changes": changes,
        "worse": len(worse),
        "_docs": docs, "_report": report, "_nb": nb,
    }


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    docs, report, nb = result["_docs"], dict(result["_report"]), result["_nb"]
    for audience, doc in docs.items():
        report.setdefault("documents", {}).setdefault(audience, {}).update(doc["stats"])
        report["documents"][audience]["coverage"] = doc["coverage"]
    report["rechecked_at"] = result["checked_at"]
    for audience, doc in docs.items():
        (out / f"{audience}.md").write_text(render_markdown(nb, doc, report), encoding="utf-8")
    (out / "docs.json").write_text(json.dumps(docs, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    (out / "verify.json").write_text(json.dumps(public, indent=1, ensure_ascii=False), encoding="utf-8")


def print_result(result: dict) -> None:
    nb = result["notebook"]
    state = "changed since the run" if result["notebook_changed"] else "unchanged since the run"
    print(f"{nb['name']} (sha256 {nb['sha256'][:12]}), {state}\n")
    print(f"{'document':<10} {'verified':>17} {'partial':>13} {'unchecked':>13} {'unverified':>13}")
    for audience, s in result["documents"].items():
        b, a = s["before"], s["after"]
        cell = lambda k: f"{b.get(k, 0)} → {a.get(k, 0)}"  # noqa: E731
        bad_b, bad_a = b.get("unverified", 0) + b.get("invalid", 0), a.get("unverified", 0) + a.get("invalid", 0)
        print(f"{audience:<10} {cell('verified'):>17} {cell('partial'):>13} {cell('unchecked'):>13} "
              f"{f'{bad_b} → {bad_a}':>13}")
    if result["changes"]:
        print(f"\n{len(result['changes'])} claims changed status ({result['worse']} got worse):")
        for c in sorted(result["changes"], key=lambda c: -RANK.get(c["after"], 9)):
            print(f"  [{c['document']} {c['id']}] {c['before']} → {c['after']}: {c['text'][:90]}")
            if c["problems"]:
                print(f"      {c['problems'][0][:140]}")
    else:
        print("\nNo claim changed status.")
    if result["notebook_changed"] and result["worse"]:
        print("\nStale: the notebook changed and some claims no longer hold. Regenerate the documents.")
