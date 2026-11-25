"""Check every claim against the notebook.

For each claim:

1. Every citation must exist: the cell is in range and the lines are inside it.
2. The evidence quote must appear in the cited lines (one line either side counts).
3. Code-like names in the claim (identifiers, quoted labels, numbers such as 5.0)
   must appear in the cited lines. A name also counts when it is an ALL-CAPS label
   spelled inside an identifier there ("FIFO" in `apply_fifo`), when the citation is inside the
   function it names, or when a variable in the cited lines holds that exact value
   and is assigned only once (`tgt` holds "fin_cash.alloc_lines").
4. Lines flagged as text addressed to AI tools can be cited by a risk, never used
   as evidence for anything else.

Statuses:
  verified    the quote and every name were found
  partial     some of it was found
  unchecked   the citation exists, but the claim names nothing that can be checked
              (its quote may be real; that alone proves nothing)
  unverified  the cited lines do not support the claim
  invalid     a citation is malformed or points outside the notebook

Invalid and unverified claims go back to the model for repair. The check is
lexical: it shows that the cited lines contain what the claim names, not that
the claim reads them correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .facts import Facts, Flag
from .notebook import Notebook, parse_citation
from .prompts import defuse

WINDOW = 1          # lines either side of a citation that count as "cited"
MAX_SPAN = 60       # a wider range is accepted but cannot be "verified"

STOP_CAPS = {
    "USD", "EUR", "INR", "SGD", "GBP", "AR", "AP", "GL", "FX", "ID", "IDS", "OK", "API", "SQL", "CSV", "JSON",
    "KPI", "DQ", "AI", "LLM", "PII", "UTC", "ETL", "UDF", "UDFS", "UI", "NA", "DBFS", "HTML", "URL", "ALL", "AND",
    "THE", "NOT", "YES", "NO", "N/A", "TBD", "ERP", "SLA", "CI", "CD", "PR", "VAT", "DSO", "WO", "UNAPP", "SUSP",
}


@dataclass
class Check:
    status: str
    problems: list[str] = field(default_factory=list)
    anchors_found: list[str] = field(default_factory=list)
    anchors_missing: list[str] = field(default_factory=list)
    evidence_found: bool | None = None


# ---- text helpers -----------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.replace("`", "").split())


def _numbers(text: str) -> set[float]:
    out = set()
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text):
        raw = m.group(0).replace(",", "")
        try:
            out.add(round(float(raw), 9))
        except ValueError:
            pass
    return out


_CITE_IN_TEXT = re.compile(r"\[?\bC\d+:L\d+(?:\s*[-–]\s*L?\d+)?\]?|\b(?:cells?|lines?|L)\s*\d+(?:\s*[-–]\s*\d+)?", re.I)


def anchors(text: str, ignore_numbers: set[float] | None = None) -> list[str]:
    """Code-like things a claim names that the cited lines should contain."""
    ignore_numbers = ignore_numbers or set()
    t = _CITE_IN_TEXT.sub(" ", text)
    found: list[str] = []
    for span in re.findall(r"`([^`]+)`", t):
        for tok in re.findall(r"[A-Za-z_][\w.]*|\d+(?:\.\d+)?", span):
            if len(tok) >= 2 or tok.isdigit():
                found.append(tok.rstrip("."))
    plain = re.sub(r"`[^`]+`", " ", t)
    for tok in re.findall(r"[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+|[A-Za-z]+_\w+|_\w+", plain):
        found.append(tok)
    for tok in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", plain):
        if tok not in STOP_CAPS:
            found.append(tok)
    for tok in re.findall(r"\b[a-z]+[A-Z]\w*\b", plain):          # camelCase: toPandas, saveAsTable
        found.append(tok)
    for tok in re.findall(r"[\"“']([^\"”'\s][^\"”']{0,28})[\"”']", plain):
        if not re.search(r"\s{2,}", tok) and len(tok.split()) <= 3:
            found.append(f'"{tok}"')
    for m in re.finditer(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)", plain):
        raw = m.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if ("." in raw or value >= 100) and round(value, 9) not in ignore_numbers:
            found.append(m.group(1))
    seen, out = set(), []
    for a in found:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _identifier_parts(window: str) -> set[str]:
    parts: set[str] = set()
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", window):
        for piece in re.split(r"_+|(?<=[a-z0-9])(?=[A-Z])", ident):
            if piece:
                parts.add(piece.lower())
    return parts


def _held_by_variable(value: str, window: str, aliases: dict[str, list[str]]) -> bool:
    return any(re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", window) for name in aliases.get(value, []))


def _anchor_in(anchor: str, window: str, window_numbers: set[float],
               aliases: dict[str, list[str]] | None = None) -> bool:
    aliases = aliases or {}
    if anchor.startswith('"'):
        inner = anchor.strip('"')
        return inner in window or inner.lower() in window.lower() or _held_by_variable(inner, window, aliases)
    if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", anchor):
        try:
            if round(float(anchor.replace(",", "")), 9) in window_numbers:
                return True
        except ValueError:
            return False
        return _held_by_variable(anchor.replace(",", ""), window, aliases)
    if re.search(rf"(?<![\w]){re.escape(anchor)}(?![\w])", window, re.I):
        return True
    if re.fullmatch(r"[A-Z]{3,}", anchor) and anchor.lower() in _identifier_parts(window):
        return True   # a label such as FIFO, spelled inside an identifier such as apply_fifo
    return _held_by_variable(anchor, window, aliases)


def _enclosing_def(lines, start: int) -> str:
    """The `def` line of the function that contains line `start`, if any.

    Citing a line inside `apply_fifo` supports a claim that names `apply_fifo`.
    """
    if start < 1 or start > len(lines):
        return ""
    target = lines[start - 1].text
    indent = len(target) - len(target.lstrip())
    for line in reversed(lines[:start]):
        text = line.text
        stripped = text.lstrip()
        if stripped.startswith(("def ", "async def ")):
            if len(text) - len(stripped) < indent or line.n == start:
                return text
        elif stripped and len(text) - len(stripped) == 0 and line.n != start:
            return ""
    return ""


# ---- per-claim check --------------------------------------------------------

def check_claim(nb: Notebook, claim: dict, ignore_numbers: set[float] | None = None,
                facts: Facts | None = None) -> Check:
    problems = list(claim.get("shape_problems") or [])
    untrusted = facts.injection_lines() if facts else set()
    aliases: dict[str, list[str]] = {}
    for name, value in (facts.literals if facts else {}).items():
        aliases.setdefault(value, []).append(name)
    untrusted_cited: list[str] = []
    windows: list[str] = []
    context: list[str] = []       # the `def` line of the function around each citation
    cell_texts: list[str] = []    # whole cited cells, for ALL-CAPS labels and constants
    cited_text: list[str] = []
    too_broad = False
    for cite in claim.get("citations", []):
        parsed = parse_citation(cite)
        if parsed is None:
            problems.append(f"'{cite}' is not an address like C29:L4")
            continue
        c, start, end = parsed
        cell = nb.cell(c)
        if cell is None:
            problems.append(f"{cite} does not exist: the notebook has {len(nb.cells)} cells")
            continue
        n = len(cell.lines)
        if n == 0:
            problems.append(f"{cite} does not exist: cell {c} is empty")
            continue
        if start < 1 or end > n:
            problems.append(f"{cite} does not exist: cell {c} has {n} lines (C{c}:L1-{n})")
            continue
        if end - start + 1 > MAX_SPAN:
            too_broad = True
        if any((c, n) in untrusted for n in range(start, end + 1)):
            untrusted_cited.append(cite)
        lines = nb.span_lines(c, max(1, start - WINDOW), min(n, end + WINDOW))
        windows.append("\n".join(l.text for l in lines))
        context.append(_enclosing_def(cell.lines, start))
        cell_texts.append(cell.source)
        cited_text.append(" | ".join(l.text.strip() for l in nb.span_lines(c, start, min(end, start + 1))))
    if problems:
        return Check("invalid", problems)
    if not windows:
        return Check("invalid", ["the claim has no citations"])

    window = defuse("\n".join(windows))
    window_norm = _norm(window)
    window_numbers = _numbers(window)

    evidence = claim.get("evidence", "") or ""
    ev = _norm(defuse(evidence)).strip("\"'“”").replace("…", "...")
    evidence_found: bool | None = None
    if ev:
        parts = [p.strip() for p in ev.split("...") if len(p.strip()) >= 3]
        if not parts:
            evidence_found = None
        else:
            evidence_found = all(p in window_norm or p.lower() in window_norm.lower() for p in parts)

    names = anchors(claim.get("text", ""), ignore_numbers)
    anchor_window = window + "\n" + "\n".join(context)
    cells_text = "\n".join(cell_texts)
    found = [a for a in names if _anchor_in(a, anchor_window, window_numbers, aliases)
             or (re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", a) and re.search(rf"['\"]{re.escape(a)}['\"]", cells_text))]
    missing = [a for a in names if a not in found]

    if evidence_found is False and not found:
        status = "unverified"
    elif evidence_found is False:
        status = "partial"
    elif evidence_found is True:
        if not names:
            status = "unchecked"
            problems.append("only the quote was checked: the claim names nothing in the code")
        elif not found:
            status = "unverified"     # the quote is real, but it is not about what the claim names
        else:
            status = "verified" if not missing else "partial"
    else:  # no usable evidence quote
        if not names:
            status = "unchecked"
        elif not missing:
            status = "verified"
        elif found:
            status = "partial"
        else:
            status = "unverified"
    if too_broad and status == "verified":
        status = "partial"
        problems.append(f"a citation spans more than {MAX_SPAN} lines; cite the exact lines")
    if untrusted_cited and claim.get("kind") != "risk":
        status = "unverified"
        problems.append(f"{', '.join(untrusted_cited)} is text addressed to AI tools: it can be reported as a "
                        "risk, but not used as evidence")

    sample = "; ".join(f"'{t[:90]}'" for t in cited_text[:2])
    if evidence_found is False:
        problems.append(f"the evidence \"{evidence[:80]}\" is not in the cited lines (they read: {sample})")
    if missing:
        problems.append(f"{', '.join(missing[:5])} {'does' if len(missing) == 1 else 'do'} not appear in the "
                        f"cited lines (they read: {sample})")
    return Check(status, problems, found, missing, evidence_found)


# ---- coverage ---------------------------------------------------------------

def _ranges(citations: list[str]) -> list[tuple[int, int, int]]:
    return [p for p in (parse_citation(c) for c in citations) if p]


def overlaps(a: list[str], b: list[str]) -> bool:
    for c1, s1, e1 in _ranges(a):
        for c2, s2, e2 in _ranges(b):
            if c1 == c2 and s1 <= e2 and s2 <= e1:
                return True
    return False


def uncovered_flags(doc: dict, flags: list[Flag]) -> list[Flag]:
    """Flags that no valid claim in the document cites."""
    cited = [c["citations"] for s in doc["sections"] for c in s["claims"]
             if c.get("status") not in {"invalid", "unverified", "dropped"}]
    return [f for f in flags if not any(overlaps(cites, f.citations) for cites in cited)]


def static_claim(nb: Notebook, flag: Flag) -> dict:
    first = parse_citation(flag.citations[0])
    evidence = nb.line_text(first[0], first[1]).strip()[:100] if first else ""
    return {
        "text": flag.detail, "kind": "risk", "citations": list(flag.citations[:8]),
        "evidence": evidence, "origin": "static", "flag": flag.id,
    }


def known_numbers(nb: Notebook) -> set[float]:
    """Numbers about the notebook itself (cell and line counts) are not anchors."""
    return {float(len(nb.cells)), float(nb.n_file_lines), float(sum(len(c.lines) for c in nb.cells))}


def coverage_report(doc: dict, facts: Facts) -> dict:
    important = facts.flags_by_severity("high", "medium")
    missing = uncovered_flags(doc, important)
    return {"flags_total": len(important), "flags_cited": len(important) - len(missing),
            "missing": [f.id for f in missing]}
