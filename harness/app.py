"""Command line and local web server (standard library only).

    python -m harness                       start the web app and open the browser
    python -m harness generate FILE         write the three documents from the command line
    python -m harness parse FILE            parse summary and static red flags (no model)
    python -m harness check                 check that the configured provider answers
    python -m harness eval RUN_DIR          score a run against eval/golden_cash_app.json
    python -m harness verify RUN_DIR        re-check a saved run against the notebook as it is now
    python -m harness redteam               run the attack notebooks in eval/redteam
    python -m harness mcp                   serve the tools to coding agents over MCP (stdio)

Provider and model: --provider / --model, HARNESS_PROVIDER / HARNESS_MODEL, or harness.toml.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import ROOT, Config, load_config
from .facts import analyze
from .notebook import MAX_BYTES, NotebookError, load_path, parse_bytes
from .pipeline import RunState, Settings, run
from .prompts import AUDIENCES
from .providers import ProviderError, make_provider
from .render import render_markdown

STATIC = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".py", ".sql", ".scala", ".r", ".ipynb", ".txt"}


def display_path(path: Path) -> str:
    """A path for reports that may be shared: relative to the working folder, else the file name."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def settings_from(config: Config, use_cache: bool = True) -> Settings:
    g = config.generation
    return Settings(
        parallel=bool(g.get("parallel", True)), cache=use_cache and bool(g.get("cache", True)),
        max_repair_rounds=int(g.get("max_repair_rounds", 2)), on_unverified=g.get("on_unverified", "flag"),
        coverage_audiences=tuple(g.get("coverage_audiences", ["agent"])),
        max_prompt_tokens=int(g.get("max_prompt_tokens", 120_000)),
        local_only=bool(config.policy.get("local_only", False)),
        runs_dir=config.path("runs_dir"), cache_dir=config.path("cache_dir"),
    )


# ---- web app ----------------------------------------------------------------

class App:
    def __init__(self, config: Config):
        self.config = config
        self.notebooks: dict[str, tuple] = {}
        self.runs: dict[str, RunState] = {}
        self.lock = threading.Lock()

    def providers(self) -> list[dict]:
        out = []
        for name in self.config.providers:
            _, cfg = self.config.provider_cfg(name)
            out.append(make_provider(name, cfg).describe())
        return out

    def load(self, payload: dict) -> dict:
        if payload.get("path"):
            p = Path(str(payload["path"]).strip().strip('"').strip("'")).expanduser()
            if p.suffix.lower() not in ALLOWED_SUFFIXES:
                raise NotebookError(f"Unsupported file type '{p.suffix}'. Use .py, .sql, .scala, .r or .ipynb.")
            nb = load_path(p)
            source = display_path(p)
        elif "content" in payload:
            data = str(payload["content"]).encode("utf-8")
            if len(data) > MAX_BYTES:
                raise NotebookError("File is too large.")
            nb = parse_bytes(data, str(payload.get("filename") or "upload.py"))
            source = None
        else:
            raise NotebookError("Give a path or choose a file.")
        facts = analyze(nb)
        nb_id = nb.sha256[:12]
        with self.lock:
            self.notebooks[nb_id] = (nb, facts, source)
        return {"notebook_id": nb_id, "notebook": nb.to_dict(), "facts": facts.to_dict()}

    def generate(self, payload: dict) -> dict:
        nb_id = payload.get("notebook_id")
        if nb_id not in self.notebooks:
            raise NotebookError("Load a notebook first.")
        nb, facts, source = self.notebooks[nb_id]
        name, cfg = self.config.provider_cfg(payload.get("provider") or None, (payload.get("model") or "").strip() or None)
        provider = make_provider(name, cfg)
        audiences = [a for a in payload.get("audiences", AUDIENCES) if a in AUDIENCES] or AUDIENCES
        state = RunState(nb, facts, provider, audiences, source=source)
        with self.lock:
            self.runs[state.id] = state
        settings = settings_from(self.config, bool(payload.get("use_cache", True)))
        threading.Thread(target=run, args=(state, settings), daemon=True).start()
        return {"run_id": state.id}


def make_handler(app: App, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "notebook-explainer"

        def log_message(self, fmt, *args):
            if "/api/runs/" not in (args[0] if args else ""):
                sys.stderr.write("  " + (fmt % args) + "\n")

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            # Blocks DNS-rebinding: a web page must not be able to drive this local server.
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"error": "unexpected Host header"})
                return False
            return True

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return
            path = urlparse(self.path).path
            if path == "/":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(unquote(path[len("/static/"):]))
            if path == "/api/config":
                return self._json(200, {"default_provider": app.config.provider, "providers": app.providers(),
                                        "config_source": app.config.source})
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                state = app.runs.get(parts[2])
                if not state:
                    return self._json(404, {"error": "unknown run"})
                if len(parts) == 4 and parts[3].endswith(".md"):
                    audience = parts[3][:-3]
                    if audience not in state.docs:
                        return self._json(404, {"error": "document not ready"})
                    md = render_markdown(state.notebook, state.docs[audience], state.report).encode("utf-8")
                    return self._send(200, md, "text/markdown; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{audience}.md"'})
                return self._json(200, state.public())
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return
            # A custom header forces a CORS preflight, which this server never approves.
            if self.headers.get("X-Harness") != "1":
                return self._json(403, {"error": "missing X-Harness header"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_BYTES * 2:
                    return self._json(413, {"error": "request too large"})
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "invalid JSON"})
            path = urlparse(self.path).path
            try:
                if path == "/api/load":
                    return self._json(200, app.load(payload))
                if path == "/api/generate":
                    return self._json(200, app.generate(payload))
            except (NotebookError, ProviderError, KeyError) as exc:
                return self._json(400, {"error": str(exc).strip("'\"")})
            self._json(404, {"error": "not found"})

        def _static(self, name: str) -> None:
            target = (STATIC / name).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

    return Handler


def serve(config: Config, open_browser: bool = True, port: int | None = None) -> None:
    host = config.server.get("host", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        print(f"Refusing to bind to {host}: this tool reads local files and must stay on 127.0.0.1.")
        sys.exit(2)
    base = int(port or config.server.get("port", 8765))
    app = App(config)
    server = None
    for p in range(base, base + 10):
        try:
            server = ThreadingHTTPServer((host, p), make_handler(app, p))
            break
        except OSError:
            continue
    if server is None:
        print(f"No free port between {base} and {base + 9}.")
        sys.exit(1)
    url = f"http://{host}:{server.server_address[1]}"
    provider = app.config.provider
    _, cfg = config.provider_cfg(provider)
    desc = make_provider(provider, cfg).describe()
    print(f"Notebook Explainer running at {url}")
    print(f"  config: {config.source}; default provider: {provider} ({desc['model']})"
          + ("" if desc["key_present"] else f"  [{desc['key_env']} not set]"))
    for note in config.notes:
        print(f"  {note}")
    print("  Ctrl+C to stop.")
    if open_browser and config.server.get("open_browser", True):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ---- command line -------------------------------------------------------------

def cmd_parse(args, config: Config) -> int:
    nb = load_path(args.file)
    facts = analyze(nb)
    s = nb.summary()
    print(f"{s['name']} · {s['format']} · sha256 {s['sha256'][:12]}")
    print(f"{s['cells']} cells · {s['cell_lines']} cell lines · {s['file_lines']} file lines · "
          + ", ".join(f"{k} {v}" for k, v in s["languages"].items()))
    if facts.parse_errors:
        print("Parse errors: " + "; ".join(facts.parse_errors))
    print(f"\nStatic red flags: {len(facts.flags)}")
    for f in facts.flags:
        print(f"  {f.severity.upper():6s} {f.title}")
    if args.facts:
        print("\n" + facts.fact_sheet())
    return 0


def cmd_generate(args, config: Config) -> int:
    nb = load_path(args.file)
    facts = analyze(nb)
    name, cfg = config.provider_cfg(args.provider, args.model)
    provider = make_provider(name, cfg)
    audiences = [a.strip() for a in args.audiences.split(",")] if args.audiences else AUDIENCES
    state = RunState(nb, facts, provider, audiences, source=display_path(Path(args.file)))
    printed = 0

    def follow():
        nonlocal printed
        while state.status in {"queued", "running"} or printed < len(state.events):
            for e in state.events[printed:]:
                who = f"[{e['audience']}] " if e["audience"] else ""
                print(f"{e['t']:6.1f}s  {e['stage']:<9} {who}{e['msg']}")
            printed = len(state.events)
            time.sleep(0.2)

    watcher = threading.Thread(target=follow, daemon=True)
    watcher.start()
    run(state, settings_from(config, not args.no_cache), Path(args.out) if args.out else None)
    watcher.join(timeout=2)
    if state.status != "done":
        print(f"\nFailed: {state.error}")
        return 1
    print(f"\nDocuments: {state.out_dir}" + (f" (copied to {args.out})" if args.out else ""))
    for a, d in state.docs.items():
        st = d["stats"]
        print(f"  {a:<9} {st['claims']} claims · {st['verified']} verified · {st['partial']} partial · "
              f"{st['unchecked']} unchecked · {st['unverified'] + st['invalid']} unverified · "
              f"{st['repaired']} repaired · {st['static_added']} added by static analysis")
    return 0


def cmd_check(args, config: Config) -> int:
    name, cfg = config.provider_cfg(args.provider, args.model)
    provider = make_provider(name, cfg)
    d = provider.describe()
    print(f"Provider {name}: kind {d['kind']}, model {d['model']}, base_url {d['base_url'] or '-'}")
    if provider.kind == "offline":
        print("OK: offline provider needs no model.")
        return 0
    if config.policy.get("local_only") and not provider.is_local:
        print(f"Refused: local_only is on, and {name} sends code to {provider.host}.")
        return 1
    if not d["key_present"]:
        print(f"Missing key: set {d['key_env']} in .env or the environment.")
        return 1
    t0 = time.time()
    try:
        comp = provider.complete("Answer with JSON only.", 'Return exactly {"ok": true}.',
                                 {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    except ProviderError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK in {time.time() - t0:.1f}s: {comp.text.strip()[:80]}" + (f" ({'; '.join(comp.notes)})" if comp.notes else ""))
    return 0


def cmd_eval(args, config: Config) -> int:
    from .evaluate import evaluate_run
    return evaluate_run(Path(args.run_dir), Path(args.golden), Path(args.readme) if args.readme else None)


def cmd_verify(args, config: Config) -> int:
    from .verify import print_result, verify, write_outputs
    result = verify(Path(args.run_dir), Path(args.notebook) if args.notebook else None)
    print_result(result)
    if args.out:
        write_outputs(result, Path(args.out))
        print(f"\nDocuments written to {args.out}")
    return 1 if result["notebook_changed"] and result["worse"] else 0


def cmd_redteam(args, config: Config) -> int:
    from .redteam import main as redteam_main
    return redteam_main()


def cmd_mcp(args, config: Config) -> int:
    from .mcp_server import main as mcp_main
    return mcp_main(["--root", args.root])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness", description="Cited documentation for notebooks.")
    parser.add_argument("--config", help="path to harness.toml")
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser("serve", help="start the web app (default)")
    p_serve.add_argument("--port", type=int)
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.add_argument("--provider")

    p_gen = sub.add_parser("generate", help="write the three documents for a notebook")
    p_gen.add_argument("file")
    p_gen.add_argument("--provider")
    p_gen.add_argument("--model")
    p_gen.add_argument("--out", help="also copy the documents to this folder, e.g. docs/generated")
    p_gen.add_argument("--audiences", help="comma separated: manager,developer,agent")
    p_gen.add_argument("--no-cache", action="store_true")

    p_parse = sub.add_parser("parse", help="parse summary and static red flags, no model")
    p_parse.add_argument("file")
    p_parse.add_argument("--facts", action="store_true", help="also print the fact sheet sent to the model")

    p_check = sub.add_parser("check", help="check that a provider answers")
    p_check.add_argument("--provider")
    p_check.add_argument("--model")

    p_eval = sub.add_parser("eval", help="score a run against a golden set")
    p_eval.add_argument("run_dir")
    p_eval.add_argument("--golden", default=str(ROOT / "eval" / "golden_cash_app.json"))
    p_eval.add_argument("--readme", help="rewrite the results table in this README from the run")

    p_verify = sub.add_parser("verify", help="re-check a saved run against the notebook as it is now")
    p_verify.add_argument("run_dir")
    p_verify.add_argument("--notebook", help="the notebook file (default: the path stored in report.json)")
    p_verify.add_argument("--out", help="write the re-checked documents to this folder")

    sub.add_parser("redteam", help="run the attack notebooks in eval/redteam")

    p_mcp = sub.add_parser("mcp", help="serve read-only tools to coding agents over MCP (stdio)")
    p_mcp.add_argument("--root", default=".", help="folder the tools may read")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.cmd in (None, "serve"):
            if getattr(args, "provider", None):
                config.provider = args.provider
            serve(config, open_browser=not getattr(args, "no_browser", False), port=getattr(args, "port", None))
            return 0
        handler = {"generate": cmd_generate, "parse": cmd_parse, "check": cmd_check, "eval": cmd_eval,
                   "verify": cmd_verify, "redteam": cmd_redteam, "mcp": cmd_mcp}[args.cmd]
        return handler(args, config)
    except (NotebookError, ProviderError, KeyError) as exc:
        print(f"Error: {str(exc).strip(chr(39))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
