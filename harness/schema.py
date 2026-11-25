"""The JSON contract between the harness and the model.

A document is a list of sections; a section is a list of claims; every claim has
at least one citation and a short evidence quote copied from the cited lines.
Free text cannot be checked, so the model must answer in this shape.

The schema below is sent to providers that support structured output and is
also written into the prompt for providers that do not. ``check_document`` then
validates the shape in code and returns precise error messages that are sent
back to the model if a repair is needed. It is hand-written (no Pydantic) to
keep the harness free of third-party dependencies.
"""

from __future__ import annotations

import json
import re

from .notebook import format_citation, parse_citation

KINDS = ["fact", "risk", "inference", "external", "question"]

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "One checkable statement."},
        "kind": {"type": "string", "enum": KINDS},
        "citations": {
            "type": "array", "minItems": 1,
            "items": {"type": "string", "description": "Address copied from the listing, e.g. C29:L4 or C26:L31-43"},
        },
        "evidence": {"type": "string", "description": "A short fragment copied exactly from one cited line."},
    },
    "required": ["text", "kind", "citations", "evidence"],
}

DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "claims": {"type": "array", "items": CLAIM_SCHEMA},
                },
                "required": ["heading", "claims"],
            },
        },
    },
    "required": ["title", "sections"],
}

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "drop": {"type": "boolean"},
                    **CLAIM_SCHEMA["properties"],
                },
                "required": ["id", "drop", "text", "kind", "citations", "evidence"],
            },
        },
    },
    "required": ["claims"],
}


class ShapeError(ValueError):
    """The model output is not a valid document. ``errors`` lists what is wrong."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:5]))
        self.errors = errors


def extract_json(text: str) -> dict:
    """Parse the model's text as one JSON object (tolerates code fences and chatter)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ShapeError(["the answer did not contain a JSON object"])
        t = t[start:end + 1]
    try:
        value = json.loads(t)
    except json.JSONDecodeError as exc:
        raise ShapeError([f"the answer is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"]) from exc
    if not isinstance(value, dict):
        raise ShapeError(["the answer must be a JSON object"])
    return value


def normalise_citations(raw) -> tuple[list[str], list[str]]:
    """Return (normalised citations, problems). Accepts strings or {cell,start,end} objects."""
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return [], ["citations must be a list"]
    good, problems = [], []
    for item in raw:
        parsed = parse_citation(item)
        if parsed is None:
            # tolerate several addresses in one string: "C4:L1, C29:L4"
            if isinstance(item, str) and "," in item:
                parts = [parse_citation(x) for x in item.split(",")]
                if all(parts):
                    good += [format_citation(*p) for p in parts]
                    continue
            problems.append(f"'{item}' is not an address like C29:L4 or C26:L31-43")
            continue
        cell, start, end = parsed
        if end < start:
            start, end = end, start
        good.append(format_citation(cell, start, end))
    return good, problems


def check_claim(raw, path: str) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"{path} must be an object"]
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{path}.text is missing")
    kind = raw.get("kind", "fact")
    if kind not in KINDS:
        kind = "fact"
    citations, problems = normalise_citations(raw.get("citations", []))
    errors += [f"{path}.citations: {p}" for p in problems]
    if not citations and not problems:
        errors.append(f"{path}.citations must contain at least one address")
    evidence = raw.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = str(evidence)
    claim = {"text": (text or "").strip(), "kind": kind, "citations": citations, "evidence": evidence.strip()}
    return claim, errors


def check_document(value: dict) -> tuple[dict, list[str]]:
    """Validate the shape. Returns (normalised document, per-claim problems).

    Raises ShapeError only when the document as a whole is unusable. Problems with
    single claims are returned so that only those claims are repaired.
    """
    errors: list[str] = []
    title = value.get("title") if isinstance(value.get("title"), str) else ""
    sections = value.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ShapeError(["'sections' must be a non-empty list"])
    out = {"title": title.strip() or "Untitled", "sections": []}
    claim_problems: list[str] = []
    for si, sec in enumerate(sections):
        if not isinstance(sec, dict):
            errors.append(f"sections[{si}] must be an object")
            continue
        heading = sec.get("heading") if isinstance(sec.get("heading"), str) else f"Section {si + 1}"
        claims_raw = sec.get("claims")
        if not isinstance(claims_raw, list):
            errors.append(f"sections[{si}].claims must be a list")
            continue
        claims = []
        for ci, raw in enumerate(claims_raw):
            claim, problems = check_claim(raw, f"sections[{si}].claims[{ci}]")
            if claim is None:
                claim_problems += problems
                continue
            claim["shape_problems"] = problems
            claim_problems += problems
            claims.append(claim)
        out["sections"].append({"heading": heading.strip(), "claims": claims})
    if errors and not any(s["claims"] for s in out["sections"]):
        raise ShapeError(errors)
    return out, errors + claim_problems
