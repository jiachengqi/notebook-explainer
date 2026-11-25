"""Score a run against a golden set of facts that a good explanation must contain.

A golden item is "found" in a document when some claim (not dropped) cites a
line that overlaps one of the item's spans AND its text contains one of the
item's keywords. This measures recall of the things that matter. Precision is
covered by the per-claim validation statuses.

With --readme, the results table in README.md (between the results markers) is
rewritten from the run, so the numbers in the README always come from a run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .validate import overlaps


def score(docs: dict, golden: list[dict]) -> dict:
    results = []
    for item in golden:
        row = {"id": item["id"], "what": item["what"]}
        for audience, doc in docs.items():
            hit = None
            for section in doc["sections"]:
                for c in section["claims"]:
                    if c.get("status") in {"dropped", "invalid"}:
                        continue
                    text = c["text"].lower()
                    if overlaps(c["citations"], item["spans"]) and any(k.lower() in text for k in item["keywords"]):
                        hit = c["id"]
                        break
                if hit:
                    break
            row[audience] = hit
        row["any"] = any(row.get(a) for a in docs)
        results.append(row)
    totals = {a: sum(1 for r in results if r.get(a)) for a in docs}
    totals["any"] = sum(1 for r in results if r["any"])
    return {"items": results, "totals": totals, "n": len(golden)}


START, END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {"manager": "Manager", "developer": "Developer", "agent": "Agent"}


def results_table(docs: dict, report: dict, result: dict) -> str:
    provider = report.get("provider") or {}
    rows = ["| Document | Claims | Verified | Partial | Unchecked | Unverified | Repaired | Added by static analysis "
            "| Golden items found |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for audience in ("manager", "developer", "agent"):
        if audience not in docs:
            continue
        st = docs[audience].get("stats", {})
        rows.append(f"| {NAMES[audience]} | {st.get('claims', 0)} | {st.get('verified', 0)} | {st.get('partial', 0)} "
                    f"| {st.get('unchecked', 0)} | {st.get('unverified', 0) + st.get('invalid', 0)} "
                    f"| {st.get('repaired', 0)} | {st.get('static_added', 0)} "
                    f"| {result['totals'].get(audience, 0)}/{result['n']} |")
    seconds = report.get("seconds")
    usage = report.get("usage") or {}
    meta = (f"Model: `{provider.get('name', '?')}` / `{provider.get('model', '?')}`. Golden items found by at least "
            f"one document: **{result['totals']['any']} of {result['n']}**.")
    if seconds and usage.get("input_tokens"):
        meta += (f" Run time {seconds:.0f} s, {usage.get('calls', 0)} model calls, "
                 f"{usage['input_tokens']:,} input and {usage.get('output_tokens', 0):,} output tokens.")
    return "\n".join(rows) + "\n\n" + meta


def update_readme(readme: Path, table: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: f"{START}\n{table}\n{END}", text), encoding="utf-8")
    return True


def evaluate_run(run_dir: Path, golden_path: Path, readme: Path | None = None) -> int:
    docs_file = run_dir / "docs.json"
    if not docs_file.is_file():
        print(f"No docs.json in {run_dir}. Pass a folder created by 'python -m harness generate'.")
        return 1
    docs = json.loads(docs_file.read_text(encoding="utf-8"))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))["items"]
    result = score(docs, golden)
    audiences = list(docs)
    width = max(len(r["id"]) for r in result["items"])
    print(f"{'item':<{width}}  " + "  ".join(f"{a[:9]:<9}" for a in audiences) + "  any")
    for r in result["items"]:
        cells = "  ".join(f"{(r.get(a) or '-'):<9}" for a in audiences)
        print(f"{r['id']:<{width}}  {cells}  {'yes' if r['any'] else 'NO'}")
    t = result["totals"]
    print("\nfound: " + ", ".join(f"{a} {t[a]}/{result['n']}" for a in audiences) + f", any {t['any']}/{result['n']}")
    (run_dir / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"written: {run_dir / 'eval.json'}")
    if readme:
        report_file = run_dir / "report.json"
        report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
        if update_readme(readme, results_table(docs, report, result)):
            print(f"updated the results table in {readme}")
        else:
            print(f"{readme} has no {START} ... {END} block; nothing changed")
            return 1
    return 0
