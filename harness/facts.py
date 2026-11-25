"""Static facts and red flags, found by reading the code (Python ``ast`` + regex).

Nothing here runs the notebook and nothing here calls a model. These are the
things code can find reliably and models tend to miss, for example a constant
that is quietly reassigned 25 cells after the config cell.

The facts are used twice:
1. as a "fact sheet" in the prompt, so the model is pointed at the right lines;
2. as a coverage checklist after generation: a high or medium red flag that the
   agent document does not cite is added by the harness itself.

`analyze` also redacts secrets in the notebook itself before anything else reads
it, so a secret cannot reach the prompt, the documents, the report or the UI.
"""

from __future__ import annotations

import ast
import builtins
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from .notebook import Cell, Notebook, format_citation, parse_citation

DATABRICKS_GLOBALS = {
    "spark", "dbutils", "display", "displayHTML", "sc", "sqlContext", "table", "sql",
    "getArgument", "spark_session", "__name__", "__file__",
}
PYSPARK_STAR_NAMES = {
    # pyspark.sql.types
    "StringType", "ArrayType", "StructType", "StructField", "IntegerType", "DoubleType", "LongType",
    "FloatType", "BooleanType", "DateType", "TimestampType", "DecimalType", "MapType", "BinaryType",
    "ShortType", "ByteType", "NullType", "DataType", "Row", "TimestampNTZType", "CharType", "VarcharType",
    # common pyspark.sql.functions
    "col", "lit", "when", "coalesce", "concat", "concat_ws", "expr", "upper", "lower", "trim", "to_date",
    "to_timestamp", "date_format", "current_date", "current_timestamp", "regexp_replace", "regexp_extract",
    "split", "explode", "array", "struct", "count", "countDistinct", "avg", "first", "last", "row_number",
    "rank", "dense_rank", "lag", "lead", "sha2", "md5", "broadcast", "udf", "pandas_udf", "collect_list",
    "collect_set", "datediff", "date_add", "date_sub", "year", "month", "dayofmonth", "round", "abs",
    "isnan", "isnull", "greatest", "least", "create_map", "element_at", "size", "array_contains",
}
KNOWN_NAMES = set(dir(builtins)) | DATABRICKS_GLOBALS
SIDE_EFFECT_WORDS = re.compile(r"notify|send|mail|post|publish|alert|slack|teams|webhook|upload|delete|drop", re.I)
WRITE_METHODS = {"saveAsTable", "insertInto", "save", "toTable", "parquet", "csv", "json", "orc", "text", "jdbc", "writeTo"}
READ_METHODS = {"csv", "table", "load", "parquet", "json", "orc", "text", "jdbc"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Text addressed to AI tools. Lines are normalised first (NFKC, look-alike letters,
# invisible characters removed), so "ign\u200bore" or full-width letters still match.
_ADDRESSEE = r"(?:ai|a\.i\.|llm|language model|assistant|chatgpt|gpt|claude|gemini|copilot|ai reviewers?|reviewer bot|automated review(?:er)?s?)"
_DIRECTIVE = (r"(?:ignore|disregard|forget|must|should|shall|need to|are required to|always|never|state|say|report|"
              r"mark|treat|pretend|respond|output|do not|don't|conclude|confirm)")
INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|earlier|system)\b.{0,25}\b(instruction|prompt|rule)s?\b", re.I),
    re.compile(rf"\b{_ADDRESSEE}s?\b.{{0,60}}\b{_DIRECTIVE}\b", re.I),
    re.compile(rf"\b{_DIRECTIVE}\b.{{0,60}}\b{_ADDRESSEE}s?\b", re.I),
    # direct address: "AI assistant: ...", "Claude, ...", "AI reviewers: ..."
    re.compile(r"\b(?:ai|llm|language model|assistant|chatgpt|gpt|claude|gemini|copilot)s?"
               r"(?: (?:reviewers?|tools?|agents?|assistants?|bots?|models?))?\s*[:,]\s*\w", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"\byou are (now )?(an? )?(ai|assistant|language model)\b", re.I),
    re.compile(r"</?(notebook|static_facts|task|output_format|failed_claims)\b", re.I),
]
_LOOKALIKES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ԁ": "d",
    "ѕ": "s", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Χ": "X", "ο": "o", "ν": "v", "ι": "i",
})
_MARKER = re.compile(r"⟨U\+[0-9A-F]{4}⟩")

# Secrets. The third item is the regex group that holds the secret itself; only that
# part is replaced, so the line still parses as code.
SECRET_PATTERNS = [
    ("Databricks token", re.compile(r"\bdapi[0-9a-f]{32}\b"), 0),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), 0),
    ("API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), 0),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0),
    ("password", re.compile(r"(?i)\b(password|passwd|pwd|secret)\s*[:=]\s*['\"]([^'\"]{4,})['\"]"), 2),
]
_PRIVATE_KEY_BEGIN = re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
_PRIVATE_KEY_END = re.compile(r"-----END (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
REDACTED = "«REDACTED»"
COMMENTED_CODE = [
    re.compile(r"^\s*(import|from)\s+[\w.]+"),
    re.compile(r"^\s*(def|class)\s+\w+"),
    re.compile(r"^\s*(for|while|if|elif|else|try|except|with|return|continue|break|pass|raise)\b.*"),
    re.compile(r"^\s*[\w.\[\]\"']+\s*(=|\+=|-=)\s*\S"),
    re.compile(r"^\s*[\w.]+\(.*\)\s*$"),
]


@dataclass
class Flag:
    id: str
    kind: str
    severity: str           # high | medium | low
    title: str              # short, for lists
    detail: str             # one sentence, used for auto-added claims
    citations: list[str]
    question: str | None = None   # what to ask the owner, if anything


@dataclass
class Facts:
    widgets: list[dict] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)
    writes: list[dict] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    externals: dict[str, list[str]] = field(default_factory=dict)
    constants: dict[str, list[dict]] = field(default_factory=dict)
    functions: dict[str, dict] = field(default_factory=dict)
    lineage: list[dict] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    redactions: dict[tuple[int, int], str] = field(default_factory=dict)   # (cell, line) -> kind of secret
    literals: dict[str, str] = field(default_factory=dict)   # name -> value, for names assigned exactly once
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["redactions"] = [f"C{c}:L{l}" for (c, l) in self.redactions]
        return d

    def flags_by_severity(self, *levels: str) -> list[Flag]:
        return [f for f in self.flags if f.severity in levels]

    def injection_lines(self) -> set[tuple[int, int]]:
        """Lines flagged as text addressed to AI tools."""
        out: set[tuple[int, int]] = set()
        for fl in self.flags:
            if fl.kind != "prompt-injection":
                continue
            for cite in fl.citations:
                parsed = parse_citation(cite)
                if parsed:
                    out.update((parsed[0], n) for n in range(parsed[1], parsed[2] + 1))
        return out

    def fact_sheet(self) -> str:
        """Compact, cited summary for the prompt."""
        out: list[str] = []
        if self.widgets:
            out.append("Parameters (widgets):")
            for w in self.widgets:
                extra = f", choices {w['choices']}" if w.get("choices") else ""
                out.append(f"- {w['name']} ({w['kind']}, default {w['default']}{extra}) at {w['cite']}")
        if self.runs:
            out.append("Code loaded from outside this notebook:")
            for r in self.runs:
                out.append(f"- %run {r['target']} at {r['cite']}")
        if self.externals:
            out.append("Names used but not defined in this notebook (do not guess what they do):")
            for name, cites in self.externals.items():
                out.append(f"- {name}: used at {', '.join(cites[:4])}")
        if self.reads:
            out.append("Reads:")
            for r in self.reads:
                out.append(f"- {r['what']} at {r['cite']}")
        if self.writes:
            out.append("Writes:")
            for w in self.writes:
                guard = f" [only when {w['guard']}]" if w.get("guard") else " [unconditional]"
                schema = " overwriteSchema=true" if w.get("overwrite_schema") else ""
                out.append(f"- {w['target']} mode={w['mode']}{schema} at {w['cite']}{guard}")
        if self.constants:
            out.append("Constants and literal settings:")
            for name, items in self.constants.items():
                vals = "; ".join(f"{i['value']} at {i['cite']}" for i in items)
                out.append(f"- {name}: {vals}")
        if self.functions:
            out.append("Functions:")
            for name, f in self.functions.items():
                if f["calls"]:
                    use = f"called at {', '.join(f['calls'][:4])}"
                elif f.get("passed_at"):
                    use = f"passed as a value at {', '.join(f['passed_at'][:4])}"
                else:
                    use = "never used"
                out.append(f"- {name} defined at {f['span']}, {use}")
        if self.flags:
            out.append("Red flags found by static analysis (verify them against the code):")
            for fl in sorted(self.flags, key=lambda f: SEVERITY_ORDER[f.severity]):
                out.append(f"- [{fl.severity}] {fl.title} ({', '.join(fl.citations[:6])})")
        return "\n".join(out)


# ---------------------------------------------------------------------------

def analyze(nb: Notebook) -> Facts:
    facts = Facts()
    _redact_secrets(nb, facts)
    _flag_hidden_characters(nb, facts)
    st = _State(nb)
    for cell in nb.cells:
        if cell.lang == "run":
            target = (cell.magic or "").split(maxsplit=1)[1] if len((cell.magic or "").split()) > 1 else ""
            facts.runs.append({"target": target, "cite": format_citation(cell.idx, 1)})
        elif cell.lang == "sql":
            _scan_sql(cell, facts)
        elif cell.lang == "python":
            _scan_python(cell, st, facts)
        _scan_text_risks(cell, facts)
        if cell.lang == "python":
            _scan_commented_code(cell, facts)
    _finish(nb, st, facts)
    return facts


class _State:
    """Accumulates names across cells (notebook cells share one global scope)."""

    def __init__(self, nb: Notebook):
        self.nb = nb
        self.bound: set[str] = set()
        self.loads: dict[str, list[str]] = {}          # name -> cites (excluding self-references)
        self.loads_in_functions: set[str] = set()
        self.module_assigns: dict[str, list[dict]] = {}  # name -> [{cite, value, literal, order}]
        self.defs: dict[str, dict] = {}
        self.name_calls: list[dict] = []                 # {name, cite, guards, cell}
        self.imports: dict[str, str] = {}                # alias -> cite
        self.star_imports: list[str] = []                # modules imported with *
        self.method_calls: list[dict] = []               # {method, cite, receiver, node_src}
        self.global_mutations: list[dict] = []
        self.print_checks: list[dict] = []
        self.exits: list[dict] = []
        self.mutable_defaults: list[dict] = []
        self.order = 0


def _cite(cell: Cell, node: ast.AST) -> str:
    return format_citation(cell.idx, node.lineno)


def _span(cell: Cell, node: ast.AST) -> str:
    return format_citation(cell.idx, node.lineno, getattr(node, "end_lineno", node.lineno))


def _literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _src(source: str, node: ast.AST, limit: int = 80) -> str:
    text = ast.get_source_segment(source, node) or ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _names_in_target(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out += _names_in_target(elt)
        return out
    if isinstance(target, ast.Starred):
        return _names_in_target(target.value)
    return []


def _scan_python(cell: Cell, st: _State, facts: Facts) -> None:
    source = "\n".join(cell.code_lines())
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        facts.parse_errors.append(f"C{cell.idx}: Python parse error at line {exc.lineno}: {exc.msg}")
        return
    _Visitor(cell, source, st, facts).visit(tree)


class _Visitor(ast.NodeVisitor):
    def __init__(self, cell: Cell, source: str, st: _State, facts: Facts):
        self.cell, self.source, self.st, self.facts = cell, source, st, facts
        self.guards: list[str] = []
        self.func: str | None = None
        self.func_locals: set[str] = set()
        self.assign_targets: set[str] = set()

    # -- scopes ---------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        st = self.st
        st.bound.add(node.name)
        if self.func is None:
            st.defs[node.name] = {"cite": _cite(self.cell, node), "span": _span(self.cell, node), "calls": []}
        params = [a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs]
        if node.args.vararg:
            params.append(node.args.vararg.arg)
        if node.args.kwarg:
            params.append(node.args.kwarg.arg)
        st.bound.update(params)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                st.mutable_defaults.append({"func": node.name, "cite": _cite(self.cell, node),
                                            "value": _src(self.source, default)})
        locals_ = set(params)
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for t in targets:
                    locals_.update(_names_in_target(t))
            elif isinstance(sub, (ast.For, ast.comprehension)):
                locals_.update(_names_in_target(sub.target))
        saved = (self.func, self.func_locals, self.guards)
        self.func, self.func_locals, self.guards = node.name, locals_, []
        for stmt in node.body:
            self.visit(stmt)
        self.func, self.func_locals, self.guards = saved

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self.st.bound.update(a.arg for a in node.args.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.st.bound.add(node.name)
        self.generic_visit(node)

    # -- guards -------------------------------------------------------------
    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        test = _src(self.source, node.test)
        self.visit(node.test)
        body_is_print = node.body and all(
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name) and s.value.func.id in {"print", "display"}
            for s in node.body
        ) and not node.orelse
        if body_is_print and self.func is None:
            self.st.print_checks.append({"cite": _span(self.cell, node), "test": test, "cell": self.cell.idx,
                                         "line": node.lineno})
        self.guards.append(test)
        for s in node.body:
            self.visit(s)
        self.guards.pop()
        if node.orelse:
            self.guards.append(f"not ({test})")
            for s in node.orelse:
                self.visit(s)
            self.guards.pop()

    # -- bindings -------------------------------------------------------------
    def _record_assign(self, targets: list[ast.AST], value: ast.AST | None, node: ast.AST) -> None:
        names = []
        for t in targets:
            names += _names_in_target(t)
        self.st.bound.update(names)
        if self.func is None and value is not None:
            for name in names:
                self.st.order += 1
                self.st.module_assigns.setdefault(name, []).append({
                    "cite": _cite(self.cell, node), "value": _src(self.source, value, 60),
                    "literal": _literal(value), "order": self.st.order,
                })
        if self.func is not None:
            for t in targets:
                if isinstance(t, ast.Subscript):
                    base = t.value
                    while isinstance(base, ast.Subscript):
                        base = base.value
                    if isinstance(base, ast.Name) and base.id not in self.func_locals:
                        self.st.global_mutations.append({"func": self.func, "name": base.id,
                                                         "cite": _cite(self.cell, node)})

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        saved = self.assign_targets
        self.assign_targets = {n for t in node.targets for n in _names_in_target(t)}
        self.visit(node.value)
        self.assign_targets = saved
        for t in node.targets:
            if not isinstance(t, ast.Name):
                self.visit(t)
        self._record_assign(node.targets, node.value, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
        self._record_assign([node.target], node.value, node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.visit(node.value)
        self._record_assign([node.target], node.value, node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.st.bound.update(_names_in_target(node.target))
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self.st.bound.update(_names_in_target(node.target))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                self.st.bound.update(_names_in_target(item.optional_vars))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.name:
            self.st.bound.add(node.name)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.st.bound.add(node.target.id)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.st.bound.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            name = (alias.asname or alias.name).split(".")[0]
            self.st.bound.add(name)
            self.st.imports.setdefault(name, _cite(self.cell, node))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "*":
                self.st.star_imports.append(node.module or "")
                continue
            name = alias.asname or alias.name
            self.st.bound.add(name)
            self.st.imports.setdefault(name, _cite(self.cell, node))

    # -- uses -----------------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            if node.id in self.assign_targets and self.func is None:
                return  # `x = x.filter(...)` is a rebuild, not a use
            self.st.loads.setdefault(node.id, []).append(_cite(self.cell, node))
            if self.func is not None and node.id not in self.func_locals:
                self.st.loads_in_functions.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        cite = _cite(self.cell, node)
        if isinstance(func, ast.Name):
            self.st.name_calls.append({"name": func.id, "cite": cite, "guards": list(self.guards),
                                       "cell": self.cell.idx, "line": node.lineno})
            if func.id in self.st.defs:
                pass
        elif isinstance(func, ast.Attribute):
            receiver = _src(self.source, func.value, 120)
            n_keys = 1
            if func.attr == "dropDuplicates" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                n_keys = len(node.args[0].elts)
            self.st.method_calls.append({"method": func.attr, "cite": cite, "receiver": receiver,
                                         "guards": list(self.guards), "cell": self.cell.idx,
                                         "line": node.lineno, "n_keys": n_keys})
            self._check_write(node, func)
            self._check_read(node, func, receiver)
            self._check_widget(node, func, receiver)
            if func.attr == "exit" and receiver.endswith("notebook"):
                self.st.exits.append({"cite": cite, "arg": _src(self.source, node, 120),
                                      "guards": list(self.guards)})
        self.generic_visit(node)

    def _chain(self, node: ast.AST) -> list[tuple[str, list[ast.AST]]]:
        chain = []
        while True:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                chain.append((node.func.attr, node.args))
                node = node.func.value
            elif isinstance(node, ast.Attribute):
                chain.append((node.attr, []))
                node = node.value
            else:
                break
        return list(reversed(chain))

    def _const_or_src(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            assigns = self.st.module_assigns.get(node.id) or []
            for a in reversed(assigns):
                if a["literal"]:
                    return a["value"].strip("'\"")
        return _src(self.source, node)

    def _check_write(self, node: ast.Call, func: ast.Attribute) -> None:
        if func.attr not in WRITE_METHODS:
            return
        chain = self._chain(node)
        attrs = [a for a, _ in chain]
        if "write" not in attrs and "writeStream" not in attrs and func.attr not in {"saveAsTable", "insertInto", "writeTo"}:
            return
        mode = "default (error if exists)"
        overwrite_schema = False
        for attr, args in chain:
            if attr == "mode" and args:
                mode = self._const_or_src(args[0])
            if attr == "option" and len(args) >= 2:
                key = self._const_or_src(args[0])
                if key.lower() == "overwriteschema" and self._const_or_src(args[1]).lower() == "true":
                    overwrite_schema = True
        target = self._const_or_src(node.args[0]) if node.args else "(unnamed)"
        target_cite = None
        if node.args and isinstance(node.args[0], ast.Name):
            literal = [a for a in self.st.module_assigns.get(node.args[0].id, []) if a["literal"]]
            target_cite = literal[-1]["cite"] if literal else None
        self.facts.writes.append({
            "target": target, "mode": mode, "overwrite_schema": overwrite_schema,
            "guard": " and ".join(self.guards) or None, "cite": _cite(self.cell, node),
            "target_cite": target_cite, "cell": self.cell.idx, "line": node.lineno,
        })

    def _check_read(self, node: ast.Call, func: ast.Attribute, receiver: str) -> None:
        if func.attr in READ_METHODS and (".read" in "." + receiver or receiver == "spark"):
            self.facts.reads.append({"what": f"`{_src(self.source, node, 110)}`", "cite": _cite(self.cell, node)})
        elif func.attr.startswith("read_") and receiver in {"pd", "pandas"}:
            self.facts.reads.append({"what": f"`{_src(self.source, node, 110)}`", "cite": _cite(self.cell, node)})
        elif func.attr == "sql" and receiver == "spark" and node.args:
            sql = self._const_or_src(node.args[0])
            for kind, table, mode in _sql_tables(sql):
                target = self.facts.writes if kind == "write" else self.facts.reads
                if kind == "write":
                    target.append({"target": table, "mode": mode, "overwrite_schema": False,
                                   "guard": " and ".join(self.guards) or None, "cite": _cite(self.cell, node),
                                   "cell": self.cell.idx, "line": node.lineno})
                else:
                    target.append({"what": f"table {table}", "cite": _cite(self.cell, node)})

    def _check_widget(self, node: ast.Call, func: ast.Attribute, receiver: str) -> None:
        if not receiver.endswith("widgets") or func.attr == "get" or func.attr.startswith("remove"):
            return
        args = [self._const_or_src(a) for a in node.args]
        choices = None
        if len(node.args) >= 3 and _literal(node.args[2]):
            choices = ast.literal_eval(node.args[2])
        self.facts.widgets.append({
            "name": args[0] if args else "?", "kind": func.attr,
            "default": repr(args[1]) if len(args) > 1 else "''",
            "choices": choices, "cite": _cite(self.cell, node),
        })


def _sql_tables(sql: str) -> list[tuple[str, str, str]]:
    """(kind, table, mode) for tables an SQL statement reads or writes."""
    out = []
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    for m in re.finditer(r"\b(insert\s+(?:into|overwrite)(?:\s+table)?|merge\s+into|update(?!\s+set\b)|delete\s+from|"
                         r"create\s+(?:or\s+replace\s+)?(?:temp\s+|temporary\s+)?(?:table|view)(?:\s+if\s+not\s+exists)?|"
                         r"drop\s+table(?:\s+if\s+exists)?)\s+([A-Za-z_`][\w.`]*)", body, re.I):
        verb = " ".join(m.group(1).lower().split())
        mode = ("overwrite" if "overwrite" in verb or "or replace" in verb else
                "append" if verb.startswith("insert into") else verb.split()[0])
        out.append(("write", m.group(2).strip("`"), mode))
    for m in re.finditer(r"\b(?:from|join)\s+([A-Za-z_`][\w.`]*)", body, re.I):
        out.append(("read", m.group(1).strip("`"), ""))
    return out


def _scan_sql(cell: Cell, facts: Facts) -> None:
    for line in cell.lines:
        text = line.text
        if text.strip().startswith("%") or text.strip().startswith("--"):
            continue
        for kind, table, mode in _sql_tables(text):
            cite = format_citation(cell.idx, line.n)
            if kind == "write":
                facts.writes.append({"target": table, "mode": mode, "overwrite_schema": False, "guard": None,
                                     "cite": cite, "cell": cell.idx, "line": line.n})
            else:
                facts.reads.append({"what": f"table {table}", "cite": cite})


def _redact_secrets(nb: Notebook, facts: Facts) -> None:
    for cell in nb.cells:
        in_key = False
        for line in cell.lines:
            label = None
            if in_key:
                if _PRIVATE_KEY_END.search(line.text):
                    in_key = False
                elif line.text.strip():
                    indent = line.text[:len(line.text) - len(line.text.lstrip())]
                    line.text, label = indent + REDACTED, "private key"
            elif _PRIVATE_KEY_BEGIN.search(line.text):
                in_key, label = True, "private key"
            for name, pattern, group in SECRET_PATTERNS:
                if pattern.search(line.text):
                    line.text = pattern.sub(lambda m: _mask(m, group), line.text)
                    label = label or name
            if label:
                facts.redactions[(cell.idx, line.n)] = label
    for (c, n), label in facts.redactions.items():
        if label == "private key" and (c, n - 1) in facts.redactions:
            continue   # one flag per key block
        cite = format_citation(c, n)
        facts.flags.append(Flag(
            id=f"secret:{cite}", kind="secret", severity="high",
            title=f"Possible hard-coded {label} at {cite}",
            detail=f"Line {cite} looks like it contains a hard-coded {label}. It is redacted everywhere the "
                   "harness shows or sends the notebook.",
            citations=[cite],
            question="Can this secret be moved to a secret scope and rotated?",
        ))


def redact_text(text: str) -> str:
    """Apply the secret patterns to any text, e.g. a model's answer, before it is shown or saved."""
    for _, pattern, group in SECRET_PATTERNS:
        text = pattern.sub(lambda m: _mask(m, group), text)
    return text


def _mask(m: re.Match, group: int) -> str:
    if group == 0:
        return REDACTED
    start, end = m.start(group) - m.start(0), m.end(group) - m.start(0)
    whole = m.group(0)
    return whole[:start] + REDACTED + whole[end:]


def _flag_hidden_characters(nb: Notebook, facts: Facts) -> None:
    for item in nb.hidden:
        cite = format_citation(item["cell"], item["line"])
        chars = ", ".join(item["chars"])
        if item["bidi"]:
            facts.flags.append(Flag(
                id=f"hidden-unicode:{cite}", kind="hidden-unicode", severity="high",
                title=f"Bidirectional control character at {cite} ({chars})",
                detail=f"Line {cite} contains {chars}, which changes how the code is displayed, so what a "
                       "reviewer reads can differ from what runs (Trojan Source, CVE-2021-42574).",
                citations=[cite],
                question="Was this character added on purpose?",
            ))
        else:
            facts.flags.append(Flag(
                id=f"hidden-unicode:{cite}", kind="hidden-unicode", severity="medium",
                title=f"Invisible character at {cite} ({chars})",
                detail=f"Line {cite} contains {chars}. Invisible characters can make two names that look "
                       "the same behave differently, or hide text from a reviewer.",
                citations=[cite],
            ))


def _normalise_for_scan(text: str) -> str:
    text = _MARKER.sub("", text)
    text = unicodedata.normalize("NFKC", text).translate(_LOOKALIKES)
    return " ".join(text.split())


def _scan_text_risks(cell: Cell, facts: Facts) -> None:
    for line in cell.lines:
        normalised = _normalise_for_scan(line.text)
        for pattern in INJECTION_PATTERNS:
            if pattern.search(normalised):
                cite = format_citation(cell.idx, line.n)
                facts.flags.append(Flag(
                    id=f"injection:{cite}", kind="prompt-injection", severity="high",
                    title=f"Text addressed to AI tools at {cite}",
                    detail=f"The text at {cite} gives instructions to AI tools. The harness treats it as data, "
                           "and a person should review it.",
                    citations=[cite],
                ))
                break


def _scan_commented_code(cell: Cell, facts: Facts) -> None:
    block: list[tuple[int, str]] = []

    def flush() -> None:
        code = [(n, t) for n, t in block if t.strip()]
        strong = any(re.match(r"^\s*([\w.\[\]\"']+\s*(=|\+=|-=)\s*\S|import |from |def )", t) for _, t in code)
        if code and (len(code) >= 3 or strong):
            start, end = block[0][0], block[-1][0]
            while block and not block[-1][1].strip():
                block.pop()
            end = block[-1][0]
            cite = format_citation(cell.idx, start, end)
            facts.flags.append(Flag(
                id=f"commented-code:{cite}", kind="dead-code", severity="low",
                title=f"Commented-out code at {cite} ({len(code)} line{'s' if len(code) != 1 else ''})",
                detail=f"Lines {cite} are commented-out code: they do not run, but they show earlier or "
                       "alternative logic.",
                citations=[cite],
            ))
        block.clear()

    for line in cell.lines:
        s = line.text.strip()
        if s.startswith("#"):
            body = s[1:]
            body = body[1:] if body.startswith(" ") else body
            if not body.strip():
                if block:
                    block.append((line.n, ""))
                continue
            if any(p.match(body) for p in COMMENTED_CODE) and not re.search(r"\b(TODO|NOTE)\b", body):
                block.append((line.n, body))
                continue
        flush()
    flush()


# ---------------------------------------------------------------------------

def _finish(nb: Notebook, st: _State, facts: Facts) -> None:
    run_cites = [r["cite"] for r in facts.runs]

    # functions and where they are called
    for name, d in st.defs.items():
        d["calls"] = [c["cite"] for c in st.name_calls if c["name"] == name]
        call_set = set(d["calls"])
        d["passed_at"] = [c for c in st.loads.get(name, []) if c not in call_set]
        facts.functions[name] = d

    # constants: UPPER_CASE names, or names assigned a literal more than once
    for name, items in st.module_assigns.items():
        literal_items = [i for i in items if i["literal"]]
        if name.isupper() or len(literal_items) >= 2:
            facts.constants[name] = [{"cite": i["cite"], "value": i["value"]} for i in items]
        if len(items) == 1 and items[0]["literal"]:
            value = items[0]["value"]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            facts.literals[name] = value

    # externals: used (loaded or called) but never bound anywhere
    used = set(st.loads) | {c["name"] for c in st.name_calls}
    star_known = PYSPARK_STAR_NAMES if any(m.startswith("pyspark") for m in st.star_imports) else set()
    externals = sorted(n for n in used if n not in st.bound and n not in KNOWN_NAMES and n not in star_known)
    for name in externals:
        cites = [c["cite"] for c in st.name_calls if c["name"] == name] or st.loads.get(name, [])
        facts.externals[name] = sorted(set(cites), key=_cite_key)

    # lineage: variables defined in one cell and used in later cells
    for name, assigns in st.module_assigns.items():
        if name in st.defs or name.isupper():
            continue
        def_cells = sorted({int(a["cite"][1:].split(":")[0]) for a in assigns})
        use_cells = sorted({int(c[1:].split(":")[0]) for c in st.loads.get(name, [])})
        later = [c for c in use_cells if c > def_cells[0]]
        if len(set(later) - set(def_cells)) >= 2:
            facts.lineage.append({"name": name, "defined": [a["cite"] for a in assigns][:6],
                                  "used_in_cells": later[:12]})
    facts.lineage.sort(key=lambda x: -len(x["used_in_cells"]))
    facts.lineage = facts.lineage[:15]

    flags = facts.flags

    # reassigned constants
    for name, items in facts.constants.items():
        if len(items) < 2:
            continue
        if not (name.isupper() or all(i["value"].replace(".", "").replace("-", "").isdigit() for i in items)):
            continue
        cites = [i["cite"] for i in items]
        first, last = items[0], items[-1]
        late = " Functions defined earlier read the global when they run, so they use the last value too." \
            if name in st.loads_in_functions else ""
        flags.append(Flag(
            id=f"reassigned-constant:{name}", kind="reassigned-constant",
            severity="high" if name.isupper() else "medium",
            title=f"`{name}` is set to {first['value']} at {first['cite']} and reassigned to {last['value']} at {last['cite']}",
            detail=f"`{name}` is set to `{first['value']}` at {first['cite']} and reassigned to `{last['value']}` at "
                   f"{last['cite']}; code that runs after {last['cite']} uses `{last['value']}`.{late}",
            citations=cites,
            question=f"Which value of `{name}` is intended: {first['value']} or {last['value']}?",
        ))

    # unused definitions
    for name, items in st.module_assigns.items():
        if name.isupper() and name not in st.loads:
            cite = items[0]["cite"]
            flags.append(Flag(
                id=f"unused-constant:{name}", kind="unused-config", severity="medium",
                title=f"`{name}` = {items[0]['value']} is set at {cite} but never used",
                detail=f"`{name}` is set to `{items[0]['value']}` at {cite} but is never read anywhere in the "
                       "notebook, so whatever rule it describes is not applied.",
                citations=[cite],
                question=f"Should `{name}` be enforced, or removed?",
            ))
    for name, d in facts.functions.items():
        if not d["calls"] and name not in st.loads:
            flags.append(Flag(
                id=f"unused-function:{name}", kind="dead-code", severity="low",
                title=f"Function `{name}` ({d['span']}) is never called",
                detail=f"Function `{name}` is defined at {d['span']} but never called in this notebook.",
                citations=[d["span"]],
            ))
    for name, cite in st.imports.items():
        if name not in st.loads and not any(c["name"] == name for c in st.name_calls):
            flags.append(Flag(id=f"unused-import:{name}", kind="dead-code", severity="low",
                              title=f"Import `{name}` at {cite} is never used",
                              detail=f"`{name}` is imported at {cite} but never used.", citations=[cite]))

    # external dependencies
    if facts.externals:
        names = ", ".join(f"`{n}`" for n in facts.externals)
        first_uses = [cites[0] for cites in facts.externals.values() if cites]
        where = (f"the only code it loads from outside is `%run {facts.runs[0]['target']}` at {facts.runs[0]['cite']}"
                 if facts.runs else "no %run or import provides them")
        if st.star_imports and not star_known:
            where += " (a `from ... import *` may also provide some of them)"
        one = len(facts.externals) == 1
        flags.append(Flag(
            id="external-names", kind="external-dependency", severity="medium",
            title=f"Uses {names} without defining {'it' if one else 'them'}",
            detail=f"{names} {'is' if one else 'are'} used but not defined in this notebook; {where}. "
                   f"{'Its' if one else 'Their'} behaviour is not visible here.",
            citations=run_cites[:1] + first_uses[:6],
            question="Can you share the code these functions come from?",
        ))

    # writes
    guards = {w["guard"] for w in facts.writes if w.get("guard")}
    for w in facts.writes:
        if w["mode"] == "overwrite":
            schema = " with `overwriteSchema=true`" if w["overwrite_schema"] else ""
            flags.append(Flag(
                id=f"overwrite:{w['cite']}", kind="overwrite-write", severity="high",
                title=f"Overwrites table {w['target']}{schema} at {w['cite']}",
                detail=f"At {w['cite']} the notebook replaces the whole table `{w['target']}` (mode `overwrite`"
                       f"{', schema may change' if w['overwrite_schema'] else ''}), so earlier contents are lost on "
                       "every run.",
                citations=[w["cite"]] + ([w["target_cite"]] if w.get("target_cite") else []),
                question=f"Should `{w['target']}` keep history, or be overwritten per run?",
            ))
        elif w["mode"] == "append":
            flags.append(Flag(
                id=f"append:{w['cite']}", kind="append-write", severity="medium",
                title=f"Appends to table {w['target']} at {w['cite']}",
                detail=f"At {w['cite']} the notebook appends to `{w['target']}`, so running it twice for the same "
                       "inputs adds the same rows twice.",
                citations=[w["cite"]] + ([w["target_cite"]] if w.get("target_cite") else []),
            ))

    # side effects outside the guard that protects the writes
    if guards:
        guard_vars = set()
        for g in guards:
            guard_vars.update(re.findall(r"[A-Za-z_]\w*", g))
        for call in st.name_calls:
            if call["name"] not in facts.externals:
                continue
            if not (SIDE_EFFECT_WORDS.search(call["name"]) or re.search(r"log|write|save|insert|update", call["name"], re.I)):
                continue
            call_guard_text = " ".join(call["guards"])
            if any(v in call_guard_text for v in guard_vars
                   if len(v) > 1 and v not in {"and", "or", "not", "None", "True", "False"}):
                continue
            noisy = bool(SIDE_EFFECT_WORDS.search(call["name"]))
            guard_text = sorted(guards)[0]
            flags.append(Flag(
                id=f"ungated-call:{call['name']}:{call['cite']}", kind="ungated-side-effect",
                severity="high" if noisy else "medium",
                title=f"`{call['name']}` at {call['cite']} runs even when the writes are skipped",
                detail=f"`{call['name']}` at {call['cite']} is not inside `if {guard_text}` like the table writes, "
                       "so it also runs during dry runs.",
                citations=[call["cite"]] + [w["cite"] for w in facts.writes if w.get("guard")][:1],
                question=f"Should `{call['name']}` run during dry runs?",
            ))

    # checks that only print, and checks after the first write
    first_write = min(((w["cell"], w["line"]) for w in facts.writes if "cell" in w), default=None)
    for chk in st.print_checks:
        after_write = first_write is not None and (chk["cell"], chk["line"]) > first_write
        flags.append(Flag(
            id=f"print-check:{chk['cite']}", kind="print-only-check",
            severity="high" if after_write else "medium",
            title=f"Check `{chk['test']}` at {chk['cite']} only prints"
                  + (" and runs after data is written" if after_write else ""),
            detail=f"The check `{chk['test']}` at {chk['cite']} only prints a message; it does not stop the run"
                   + (f", and it runs after the first table write at C{first_write[0]}:L{first_write[1]}."
                      if after_write else "."),
            citations=[chk["cite"]] + ([format_citation(*first_write)] if after_write else []),
        ))

    # exit status
    for ex in st.exits:
        if not ex["guards"] and re.search(r"['\"]ok['\"]", ex["arg"], re.I):
            flags.append(Flag(
                id=f"constant-exit:{ex['cite']}", kind="constant-exit", severity="high",
                title=f"Always exits with status ok at {ex['cite']}",
                detail=f"At {ex['cite']} the notebook always exits with status \"ok\", whatever its checks found, "
                       "so a job that runs it reports success.",
                citations=[ex["cite"]],
                question="Should the job fail when a reconciliation or data-quality check fails?",
            ))

    # global state mutated inside functions
    by_name: dict[str, list[dict]] = {}
    for m in st.global_mutations:
        by_name.setdefault(m["name"], []).append(m)
    for name, muts in by_name.items():
        funcs = list(dict.fromkeys(m["func"] for m in muts))
        cites: list[str] = []
        for f in funcs[:4]:
            if f in facts.functions:
                cites.append(facts.functions[f]["cite"])
            cites.append(next(m["cite"] for m in muts if m["func"] == f))
        flags.append(Flag(
            id=f"global-state:{name}", kind="global-state", severity="medium",
            title=f"Global `{name}` is modified inside {', '.join(f'`{f}`' for f in funcs)}",
            detail=f"Functions {', '.join(f'`{f}`' for f in funcs)} modify the global `{name}`, so results depend "
                   "on running the cells in order, once.",
            citations=cites,
        ))

    # method-call patterns
    def calls(method: str, receiver_hint: str | None = None) -> list[dict]:
        return [c for c in st.method_calls if c["method"] == method
                and (receiver_hint is None or receiver_hint in c["receiver"])]

    driver = calls("toPandas") + calls("collect") + calls("iterrows") + calls("itertuples")
    if driver:
        names = sorted({c["method"] for c in driver})
        flags.append(Flag(
            id="driver-side", kind="scalability", severity="medium",
            title=f"Data is pulled to the driver ({', '.join(names)})",
            detail=f"The notebook moves data to the driver and processes it row by row ({', '.join(names)} at "
                   f"{', '.join(c['cite'] for c in driver[:4])}), which limits how far it scales.",
            citations=[c["cite"] for c in driver][:6],
        ))
    nondet = calls("dropDuplicates") + [c for c in calls("first") if c["receiver"] in {"F", "functions"}]
    for c in nondet:
        what = "keeps an arbitrary row per key" if c["method"] == "dropDuplicates" else "picks an arbitrary value"
        severity = "low" if c["method"] == "dropDuplicates" and c.get("n_keys", 1) > 1 else "medium"
        flags.append(Flag(
            id=f"nondeterministic:{c['cite']}", kind="non-deterministic", severity=severity,
            title=f"`{c['method']}` at {c['cite']} {what}",
            detail=f"`{c['method']}` at {c['cite']} {what}, so the result can differ between runs when keys repeat.",
            citations=[c["cite"]],
        ))
    udfs = [c for c in calls("udf") if c["receiver"] in {"F", "functions"}] + \
        [c for c in st.name_calls if c["name"] in {"udf", "pandas_udf"}]
    if udfs:
        flags.append(Flag(id="python-udf", kind="performance", severity="low",
                          title="Python UDFs are used",
                          detail=f"Python UDFs are registered at {', '.join(c['cite'] for c in udfs[:4])}; they are "
                                 "slower than built-in Spark functions.",
                          citations=[c["cite"] for c in udfs][:4]))
    for c in calls("set", "conf"):
        flags.append(Flag(id=f"spark-conf:{c['cite']}", kind="config", severity="low",
                          title=f"Spark configuration is set in code at {c['cite']}",
                          detail=f"The notebook sets a Spark configuration at {c['cite']}.", citations=[c["cite"]]))
    for m in st.mutable_defaults:
        flags.append(Flag(id=f"mutable-default:{m['func']}", kind="code-smell", severity="low",
                          title=f"`{m['func']}` has a mutable default argument {m['value']}",
                          detail=f"`{m['func']}` ({m['cite']}) uses a mutable default argument `{m['value']}`.",
                          citations=[m["cite"]]))
    for w in facts.widgets:
        if "/mnt/" in w["default"] or "dbfs:" in w["default"]:
            flags.append(Flag(id=f"mount-path:{w['cite']}", kind="config", severity="low",
                              title=f"Default path {w['default']} at {w['cite']} is a DBFS mount",
                              detail=f"The default input path {w['default']} at {w['cite']} is a DBFS mount path.",
                              citations=[w["cite"]]))

    # de-duplicate by id, sort by severity then position
    seen, unique = set(), []
    for f in flags:
        if f.id not in seen:
            seen.add(f.id)
            unique.append(f)
    unique.sort(key=lambda f: (SEVERITY_ORDER[f.severity], _cite_key(f.citations[0]) if f.citations else (0, 0)))
    facts.flags = unique


def _cite_key(cite: str) -> tuple[int, int]:
    m = re.match(r"C(\d+):L(\d+)", cite)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
