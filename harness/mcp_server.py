"""MCP server over stdio, so coding agents can use the harness while they work.

    python -m harness mcp [--root DIR]

Tools (read-only, no model calls):

    red_flags(path)                         static red flags with their citations
    explain_cell(path, cell, run_dir?)      a cell's lines with addresses, its red flags, claims citing it
    claims(run_dir, document?, status?)     claims from a saved run
    verify(run_dir, path?)                  re-check a saved run against the notebook as it is now
    check_change(before, after, run_dir?)   what an edit touches: changed lines, red flags and claims on them

Paths must be inside --root (default: the working folder) and notebooks must have
a notebook extension, so the server cannot be used to read other files.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from .facts import analyze
from .notebook import NotebookError, format_citation, load_path
from .validate import overlaps
from .verify import verify

PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
NOTEBOOK_SUFFIXES = {".py", ".sql", ".scala", ".r", ".ipynb", ".txt"}

TOOLS = [
    {"name": "red_flags",
     "description": "Static red flags in a notebook (reassigned constants, overwrites, prompt injection, "
                    "secrets, hidden Unicode...), each with C<cell>:L<line> citations. No model is called.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "explain_cell",
     "description": "One cell's lines with their addresses and file lines, the red flags in it, and, if "
                    "run_dir is given, the documented claims that cite it.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "cell": {"type": "integer"},
                                                      "run_dir": {"type": "string"}},
                     "required": ["path", "cell"]}},
    {"name": "claims",
     "description": "Claims from a saved run (a runs/ folder or docs/generated), optionally filtered by "
                    "document (manager, developer, agent) and status.",
     "inputSchema": {"type": "object", "properties": {"run_dir": {"type": "string"},
                                                      "document": {"type": "string"},
                                                      "status": {"type": "string"}},
                     "required": ["run_dir"]}},
    {"name": "verify",
     "description": "Re-check every claim of a saved run against the notebook as it is now. Lists the "
                    "claims that no longer hold. Call it after editing a notebook.",
     "inputSchema": {"type": "object", "properties": {"run_dir": {"type": "string"}, "path": {"type": "string"}},
                     "required": ["run_dir"]}},
    {"name": "check_change",
     "description": "Compare two versions of a notebook and list the changed lines, the red flags on them, "
                    "and (with run_dir) the documented claims and invariants that cite them. Call it before "
                    "committing an edit.",
     "inputSchema": {"type": "object", "properties": {"before": {"type": "string"}, "after": {"type": "string"},
                                                      "run_dir": {"type": "string"}},
                     "required": ["before", "after"]}},
]


class ToolError(Exception):
    pass


class Server:
    def __init__(self, root: Path):
        self.root = root.resolve()

    # ---- paths ------------------------------------------------------------
    def _inside(self, value: str) -> Path:
        p = Path(value).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ToolError(f"{value} is outside {self.root}") from None
        return p

    def notebook(self, value: str) -> Path:
        p = self._inside(value)
        if p.suffix.lower() not in NOTEBOOK_SUFFIXES or not p.is_file():
            raise ToolError(f"{value} is not a notebook file")
        return p

    def run_dir(self, value: str) -> Path:
        p = self._inside(value)
        if not (p / "docs.json").is_file():
            raise ToolError(f"{value} is not a run folder (no docs.json)")
        return p

    # ---- tools ------------------------------------------------------------
    def red_flags(self, path: str) -> dict:
        nb = load_path(self.notebook(path))
        facts = analyze(nb)
        return {"notebook": nb.summary(),
                "flags": [{"id": f.id, "severity": f.severity, "title": f.title, "citations": f.citations,
                           "question": f.question} for f in facts.flags]}

    def explain_cell(self, path: str, cell: int, run_dir: str | None = None) -> dict:
        nb = load_path(self.notebook(path))
        facts = analyze(nb)
        c = nb.cell(int(cell))
        if c is None:
            raise ToolError(f"cell {cell} does not exist: the notebook has {len(nb.cells)} cells")
        span = [format_citation(c.idx, 1, max(1, len(c.lines)))]
        out = {"cell": c.idx, "language": c.lang, "title": c.title,
               "lines": [{"address": format_citation(c.idx, l.n), "file_line": l.file_line, "text": l.text}
                         for l in c.lines],
               "red_flags": [{"severity": f.severity, "title": f.title, "citations": f.citations}
                             for f in facts.flags if overlaps(f.citations, span)]}
        if run_dir:
            out["claims"] = [x for x in self._claims(run_dir) if overlaps(x["citations"], span)]
        return out

    def claims(self, run_dir: str, document: str | None = None, status: str | None = None) -> dict:
        items = [x for x in self._claims(run_dir)
                 if (not document or x["document"] == document) and (not status or x["status"] == status)]
        return {"count": len(items), "claims": items}

    def verify(self, run_dir: str, path: str | None = None) -> dict:
        result = verify(self.run_dir(run_dir), self.notebook(path) if path else None)
        return {k: v for k, v in result.items() if not k.startswith("_")}

    def check_change(self, before: str, after: str, run_dir: str | None = None) -> dict:
        old, new = load_path(self.notebook(before)), load_path(self.notebook(after))
        changed: list[str] = []
        for i in range(max(len(old.cells), len(new.cells))):
            a = [l.text for l in old.cells[i].lines] if i < len(old.cells) else []
            b = [l.text for l in new.cells[i].lines] if i < len(new.cells) else []
            if a == b:
                continue
            if not a:
                changed.append(f"cell {i + 1} added")
                continue
            for tag, i1, i2, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
                if tag != "equal":
                    changed.append(format_citation(i + 1, i1 + 1, max(i1 + 1, i2)))
        cites = [c for c in changed if c.startswith("C")]
        facts = analyze(old)
        out = {"changed": changed,
               "red_flags_touched": [{"severity": f.severity, "title": f.title, "citations": f.citations}
                                     for f in facts.flags if overlaps(f.citations, cites)]}
        if run_dir:
            out["claims_touched"] = [x for x in self._claims(run_dir) if overlaps(x["citations"], cites)]
        return out

    def _claims(self, run_dir: str) -> list[dict]:
        docs = json.loads((self.run_dir(run_dir) / "docs.json").read_text(encoding="utf-8"))
        return [{"document": a, "id": c.get("id"), "status": c.get("status"), "kind": c.get("kind"),
                 "section": s.get("heading"), "text": c["text"], "citations": c["citations"]}
                for a, d in docs.items() for s in d["sections"] for c in s["claims"]
                if c.get("status") != "dropped"]

    # ---- protocol ---------------------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if mid is None:
            return None                                   # notification
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _ok(mid, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": "notebook-explainer", "version": "1.0.0"},
                             "instructions": "Read-only tools for Databricks notebooks. Call red_flags before "
                                             "editing a notebook, and check_change or verify after."})
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": TOOLS})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in {t["name"] for t in TOOLS}:
                return _err(mid, -32602, f"unknown tool {name}")
            try:
                result = getattr(self, name)(**args)
                return _ok(mid, {"content": [{"type": "text", "text": json.dumps(result, indent=1,
                                                                                ensure_ascii=False)}],
                                 "isError": False})
            except (ToolError, NotebookError, TypeError, ValueError) as exc:
                return _ok(mid, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
        return _err(mid, -32601, f"method not found: {method}")


def _ok(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness mcp")
    parser.add_argument("--root", default=".", help="folder the tools may read (default: the working folder)")
    args = parser.parse_args(argv)
    server = Server(Path(args.root))
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = _err(None, -32700, "parse error")
        else:
            reply = server.handle(msg) if isinstance(msg, dict) else _err(None, -32600, "invalid request")
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0
