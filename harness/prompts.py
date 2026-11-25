"""Assemble prompts. The texts live in harness/prompts/*.md so they can be reviewed and versioned.

Order inside the user message: the long material first (the numbered notebook,
then the static facts), and the task and output format last.

The notebook is wrapped in a tag that ends with a code derived from the file's
hash, e.g. <notebook-1a2b3c4d5e6f>. A notebook cannot contain its own closing tag
without knowing its own hash, so it cannot end the block early and pose as the
task. Tag-like text inside notebook content is also defused.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .facts import Facts
from .notebook import Notebook

PROMPT_DIR = Path(__file__).parent / "prompts"
AUDIENCES = ["manager", "developer", "agent"]

_SHAPE = ('{"title": "...", "sections": [{"heading": "...", "claims": [{"text": "...", '
          '"kind": "fact|risk|inference|external|question", "citations": ["C4:L1", "C26:L31-43"], '
          '"evidence": "fragment copied exactly from a cited line"}]}]}')


def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def system_prompt() -> str:
    return load("system")


_TAGS = re.compile(r"<(/?)(notebook|static_facts|task|output_format|failed_claims|note)\b", re.I)


def defuse(text: str) -> str:
    """Make tag-like text inert, so untrusted content cannot open or close a prompt section."""
    return _TAGS.sub(lambda m: f"‹{m.group(1)}{m.group(2)}", text)


def _notebook_block(nb: Notebook, facts: Facts) -> str:
    tag = f"notebook-{nb.sha256[:12]}"
    name = re.sub(r"[^\w.\- ]", "_", nb.name)
    return f'<{tag} name="{name}" cells="{len(nb.cells)}">\n{defuse(nb.listing())}\n</{tag}>'


def document_prompt(nb: Notebook, facts: Facts, audience: str) -> str:
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience}")
    return "\n\n".join([
        _notebook_block(nb, facts),
        f"<static_facts>\n{defuse(facts.fact_sheet())}\n</static_facts>",
        f"<task>\n{load(audience)}\n</task>",
        f"<output_format>\nReturn one JSON object with this shape:\n{_SHAPE}\n</output_format>",
    ])


def repair_prompt(nb: Notebook, facts: Facts, failed: list[dict]) -> str:
    items = [{"id": c["id"], "text": c["text"], "kind": c["kind"], "citations": c["citations"],
              "evidence": c.get("evidence", ""), "problems": c.get("problems", [])} for c in failed]
    shape = ('{"claims": [{"id": "a7", "drop": false, "text": "...", "kind": "risk", '
             '"citations": ["C29:L4"], "evidence": "TOL = 5.0"}]}')
    return "\n\n".join([
        _notebook_block(nb, facts),
        f"<failed_claims>\n{defuse(json.dumps(items, indent=1, ensure_ascii=False))}\n</failed_claims>",
        f"<task>\n{load('repair')}\n</task>",
        f"<output_format>\n{shape}\n</output_format>",
    ])


def prompt_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
