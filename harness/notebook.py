"""Parse a notebook file into cells and lines with stable addresses.

Every line gets an address ``C{cell}:L{line}``:

* cells are numbered from 1 in file order (a Databricks cell starts after each
  ``# COMMAND ----------`` separator);
* lines are numbered from 1 inside each cell, after removing the blank padding
  lines that the Databricks export adds around separators and after stripping
  the ``# MAGIC `` prefix. This matches what a reader sees in the Databricks UI,
  where line numbers restart in every cell.

For source files we also keep the line number in the file, because a coding agent
edits the file, not the notebook UI.

Supported inputs: Databricks source exports (Python, SQL, Scala, R), Jupyter
``.ipynb`` (outputs are ignored and never sent to a model), "percent" scripts
(``# %%`` cells) and plain scripts (one cell).

Invisible and bidirectional Unicode characters are replaced with a visible marker
such as ``⟨U+202E⟩`` at load time, so the model, the documents and the reader all
see the same text (see Trojan Source, CVE-2021-42574).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024

_HEADER = re.compile(r"^\s*(#|--|//)\s*Databricks notebook source\s*$")
_MAGIC_LANG = {
    "%md": "md", "%md-sandbox": "md", "%sql": "sql", "%python": "python",
    "%scala": "scala", "%r": "r", "%sh": "sh", "%run": "run", "%pip": "pip",
    "%fs": "fs", "%%sql": "sql", "%%sh": "sh", "%%bash": "sh", "%%md": "md",
}
_DEFAULT_LANG = {"#": "python", "--": "sql", "//": "scala"}

BIDI_CHARS = {
    0x202A: "LEFT-TO-RIGHT EMBEDDING", 0x202B: "RIGHT-TO-LEFT EMBEDDING", 0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE", 0x202E: "RIGHT-TO-LEFT OVERRIDE", 0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE", 0x2068: "FIRST STRONG ISOLATE", 0x2069: "POP DIRECTIONAL ISOLATE",
    0x200E: "LEFT-TO-RIGHT MARK", 0x200F: "RIGHT-TO-LEFT MARK", 0x061C: "ARABIC LETTER MARK",
}
ZERO_WIDTH_CHARS = {
    0x200B: "ZERO WIDTH SPACE", 0x200C: "ZERO WIDTH NON-JOINER", 0x200D: "ZERO WIDTH JOINER",
    0x2060: "WORD JOINER", 0xFEFF: "ZERO WIDTH NO-BREAK SPACE",
}
_HIDDEN = re.compile("[" + "".join(chr(c) for c in {**BIDI_CHARS, **ZERO_WIDTH_CHARS}) + "]")


class NotebookError(ValueError):
    """The file cannot be read as a notebook. The message is shown to the user."""


@dataclass
class Line:
    n: int                   # line number inside the cell, from 1
    file_line: int | None    # line number in the source file (None for .ipynb)
    text: str                # text as shown in the notebook UI


@dataclass
class Cell:
    idx: int                 # cell number, from 1
    lang: str                # python | sql | scala | r | md | run | sh | pip | fs | other
    lines: list[Line] = field(default_factory=list)
    title: str | None = None
    magic: str | None = None  # first-line magic command, e.g. "%run ../shared/fin_utils"

    @property
    def source(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def code_lines(self) -> list[str]:
        """Line texts with magic/shell lines blanked, so language parsers keep numbering."""
        out = []
        for line in self.lines:
            stripped = line.text.lstrip()
            out.append("" if stripped.startswith(("%", "!")) else line.text)
        return out


@dataclass
class Notebook:
    name: str
    fmt: str                 # databricks-python | databricks-sql | ... | ipynb | percent | script
    default_lang: str
    cells: list[Cell]
    sha256: str
    n_file_lines: int
    hidden: list[dict] = field(default_factory=list)   # lines that contained invisible characters

    # ---- lookup -----------------------------------------------------------
    def cell(self, idx: int) -> Cell | None:
        if 1 <= idx <= len(self.cells):
            return self.cells[idx - 1]
        return None

    def span_lines(self, cell_idx: int, start: int, end: int) -> list[Line]:
        cell = self.cell(cell_idx)
        if cell is None:
            return []
        return [line for line in cell.lines if start <= line.n <= end]

    def line_text(self, cell_idx: int, line_no: int) -> str:
        lines = self.span_lines(cell_idx, line_no, line_no)
        return lines[0].text if lines else ""

    # ---- views ------------------------------------------------------------
    def listing(self) -> str:
        """Numbered listing for the model: every line carries its address."""
        out = []
        for cell in self.cells:
            head = f"=== Cell {cell.idx} · {cell.lang}"
            if cell.title:
                head += f" · title: {cell.title}"
            if not cell.lines:
                head += " · (empty)"
            out.append(head + " ===")
            for line in cell.lines:
                out.append(f"[C{cell.idx}:L{line.n}] {line.text}")
        return "\n".join(out)

    def summary(self) -> dict:
        langs: dict[str, int] = {}
        for cell in self.cells:
            langs[cell.lang] = langs.get(cell.lang, 0) + 1
        return {
            "name": self.name,
            "format": self.fmt,
            "sha256": self.sha256,
            "cells": len(self.cells),
            "cell_lines": sum(len(c.lines) for c in self.cells),
            "file_lines": self.n_file_lines,
            "languages": langs,
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "cell_list": [
                {
                    "idx": c.idx, "lang": c.lang, "title": c.title, "magic": c.magic,
                    "lines": [{"n": l.n, "file_line": l.file_line, "text": l.text} for l in c.lines],
                }
                for c in self.cells
            ],
        }


# ---- entry points ---------------------------------------------------------

def load_path(path: str | Path) -> Notebook:
    p = Path(path).expanduser()
    if not p.is_file():
        raise NotebookError(f"File not found: {p}")
    if p.stat().st_size > MAX_BYTES:
        raise NotebookError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB: {p.name}")
    return parse_bytes(p.read_bytes(), p.name)


def parse_bytes(data: bytes, name: str) -> Notebook:
    if len(data) > MAX_BYTES:
        raise NotebookError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB: {name}")
    if data[:2] == b"PK":
        raise NotebookError(
            f"{name} looks like a zip or .dbc archive. Export the notebook from Databricks "
            "as 'Source file' or 'IPython Notebook' and load that instead."
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NotebookError(f"{name} is not UTF-8 text ({exc}).") from exc
    return parse_text(text, name, hashlib.sha256(data).hexdigest())


def parse_text(text: str, name: str, sha256: str | None = None) -> Notebook:
    sha256 = sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return _reveal_hidden(_parse(text, name, sha256))


def _parse(text: str, name: str, sha256: str) -> Notebook:
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]

    stripped = text.lstrip()
    if name.lower().endswith(".ipynb") or (stripped.startswith("{") and '"cells"' in stripped[:5000]):
        return _parse_ipynb(text, name, sha256)

    first = next((line for line in raw_lines if line.strip()), "")
    match = _HEADER.match(first)
    if match:
        return _parse_databricks(raw_lines, name, sha256, match.group(1))
    if any(re.match(r"^#\s*%%", line) for line in raw_lines):
        return _parse_percent(raw_lines, name, sha256)
    if name.lower().endswith((".py", ".sql", ".scala", ".r", ".txt")) or not stripped.startswith("{"):
        lang = {".sql": "sql", ".scala": "scala", ".r": "r"}.get(Path(name).suffix.lower(), "python")
        if not raw_lines:
            raise NotebookError(f"{name} is empty.")
        lines = [Line(i, i, t) for i, t in enumerate(raw_lines, 1)]
        return Notebook(name, "script", lang, [Cell(1, lang, lines)], sha256, len(raw_lines))
    raise NotebookError(
        f"Could not recognise {name}. Expected a Databricks source export (first line "
        "'# Databricks notebook source'), a Jupyter .ipynb, or a Python/SQL script."
    )


def reveal_hidden(text: str) -> str:
    """Replace invisible and bidirectional characters with a visible marker like ⟨U+202E⟩."""
    return _HIDDEN.sub(lambda m: f"⟨U+{ord(m.group(0)):04X}⟩", text)


def _reveal_hidden(nb: Notebook) -> Notebook:
    for cell in nb.cells:
        for line in cell.lines:
            found = _HIDDEN.findall(line.text)
            if not found:
                continue
            names = []
            for ch in dict.fromkeys(found):
                cp = ord(ch)
                names.append(f"U+{cp:04X} {BIDI_CHARS.get(cp) or ZERO_WIDTH_CHARS.get(cp)}")
            nb.hidden.append({"cell": cell.idx, "line": line.n, "chars": names,
                              "bidi": any(ord(ch) in BIDI_CHARS for ch in found)})
            line.text = reveal_hidden(line.text)
    return nb


# ---- formats --------------------------------------------------------------

def _parse_databricks(raw_lines: list[str], name: str, sha256: str, prefix: str) -> Notebook:
    default_lang = _DEFAULT_LANG[prefix]
    if prefix == "#" and name.lower().endswith(".r"):
        default_lang = "r"
    p = re.escape(prefix)
    sep = re.compile(rf"^\s*{p}\s*COMMAND\s*-{{3,}}\s*$")
    magic = re.compile(rf"^{p} MAGIC ?(.*)$")
    title = re.compile(rf"^{p} DBTITLE \d+,(.*)$")

    header_seen = False
    blocks: list[list[tuple[int, str]]] = [[]]
    for file_no, raw in enumerate(raw_lines, 1):
        if not header_seen:
            if _HEADER.match(raw):
                header_seen = True
            continue
        if sep.match(raw):
            blocks.append([])
        else:
            blocks[-1].append((file_no, raw))

    cells: list[Cell] = []
    for idx, block in enumerate(blocks, 1):
        cell_title = None
        rows: list[tuple[int, str, bool]] = []
        for file_no, raw in block:
            t = title.match(raw)
            if t and not any(r[1].strip() for r in rows):
                cell_title = t.group(1).strip() or None
                continue
            m = magic.match(raw)
            rows.append((file_no, m.group(1) if m else raw, bool(m)))
        while rows and not rows[0][1].strip():
            rows.pop(0)
        while rows and not rows[-1][1].strip():
            rows.pop()
        lines = [Line(n, file_no, text) for n, (file_no, text, _) in enumerate(rows, 1)]
        lang, magic_cmd = default_lang, None
        if rows and rows[0][2]:
            first = rows[0][1].strip()
            token = first.split()[0] if first else ""
            if token in _MAGIC_LANG:
                lang, magic_cmd = _MAGIC_LANG[token], first
            elif token.startswith("%"):
                lang, magic_cmd = "other", first
        cells.append(Cell(idx, lang, lines, cell_title, magic_cmd))

    fmt = f"databricks-{default_lang}"
    return Notebook(name, fmt, default_lang, cells, sha256, len(raw_lines))


def _parse_ipynb(text: str, name: str, sha256: str) -> Notebook:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NotebookError(f"{name} is not valid notebook JSON ({exc.msg}, line {exc.lineno}).") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise NotebookError(f"{name} has no 'cells' list, so it is not a Jupyter notebook.")
    meta = data.get("metadata") or {}
    lang = (
        (meta.get("application/vnd.databricks.v1+notebook") or {}).get("language")
        or (meta.get("kernelspec") or {}).get("language")
        or (meta.get("language_info") or {}).get("name")
        or "python"
    ).lower()

    cells: list[Cell] = []
    for idx, raw in enumerate(data["cells"], 1):
        src = raw.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        texts = src.replace("\r\n", "\n").split("\n")
        while texts and not texts[-1].strip():
            texts.pop()
        lines = [Line(n, None, t) for n, t in enumerate(texts, 1)]
        ctype = raw.get("cell_type", "code")
        cell_lang, magic_cmd = lang, None
        if ctype == "markdown":
            cell_lang = "md"
        elif ctype != "code":
            cell_lang = "other"
        elif texts and texts[0].strip().startswith("%"):
            first = texts[0].strip()
            token = first.split()[0]
            cell_lang, magic_cmd = _MAGIC_LANG.get(token, "other"), first
        title = ((raw.get("metadata") or {}).get("application/vnd.databricks.v1+cell") or {}).get("title")
        cells.append(Cell(idx, cell_lang, lines, title or None, magic_cmd))
    if not cells:
        raise NotebookError(f"{name} has no cells.")
    return Notebook(name, "ipynb", lang, cells, sha256, 0)


def _parse_percent(raw_lines: list[str], name: str, sha256: str) -> Notebook:
    cells: list[Cell] = []
    current: list[tuple[int, str]] = []
    current_lang = "python"

    def flush() -> None:
        rows = list(current)
        while rows and not rows[0][1].strip():
            rows.pop(0)
        while rows and not rows[-1][1].strip():
            rows.pop()
        if rows or cells:
            lines = [Line(n, f, t) for n, (f, t) in enumerate(rows, 1)]
            cells.append(Cell(len(cells) + 1, current_lang, lines))

    for file_no, raw in enumerate(raw_lines, 1):
        if re.match(r"^#\s*%%", raw):
            if current:
                flush()
            current = []
            current_lang = "md" if "[markdown]" in raw else "python"
        else:
            current.append((file_no, raw))
    flush()
    return Notebook(name, "percent", "python", cells, sha256, len(raw_lines))


# ---- citations ------------------------------------------------------------

_CITE = re.compile(r"^\[?\s*C\s*(\d+)\s*:\s*L\s*(\d+)\s*(?:[-–—]\s*L?\s*(\d+))?\s*\]?$", re.IGNORECASE)


def parse_citation(value) -> tuple[int, int, int] | None:
    """Parse 'C29:L4', 'C26:L31-43', '[C26:L31-L43]' or {'cell':..,'start':..,'end':..}."""
    if isinstance(value, dict):
        try:
            cell = int(value.get("cell"))
            start = int(value.get("start", value.get("line")))
            end = int(value.get("end", start))
            return cell, start, end
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    m = _CITE.match(value.strip())
    if not m:
        return None
    cell, start = int(m.group(1)), int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return cell, start, end


def format_citation(cell: int, start: int, end: int | None = None) -> str:
    if end is None or end == start:
        return f"C{cell}:L{start}"
    return f"C{cell}:L{start}-{end}"
