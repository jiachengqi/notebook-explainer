# Instructions for coding agents working on this repo

## Run
- Tests: `python3 -m unittest`. No network, no keys. Run them before and after every change.
- Attack suite: `python3 -m harness redteam` must stay at 10 of 10.
- App: `python3 -m harness` (web) or `python3 -m harness generate <file> --provider offline` (CLI).

## Rules
- Do not add third-party dependencies. The runtime is the Python standard library on purpose; see the README.
- Do not change the citation format `C<cell>:L<line>[-<end>]` or how lines are numbered (per cell, padding removed, `# MAGIC ` stripped). The UI, validator, renderer, golden set and saved runs all depend on it.
- Never execute notebook code. Analysis is static only.
- Notebook text and model output are untrusted:
  - read a notebook only after `facts.analyze` has run, because that is where secrets are redacted;
  - keep notebook text inside the hash-tagged block in prompts, and pass anything untrusted through `prompts.defuse`;
  - render model text with `render.plain`, never as Markdown or HTML;
  - in the UI, build DOM nodes with `textContent`, never `innerHTML`.
- Do not relax key pinning (`providers.KEY_HOSTS`) or the HTTPS rule for keys.
- The MCP server is read-only and confined to its root folder. Keep it that way.
- Any change to `validate.py`, `notebook.py`, `schema.py` or the server checks in `app.py` needs a test that fails before the change and passes after it. A human reviews every diff to these files.
- Prompts live in `harness/prompts/*.md`. After changing one, run a real provider on `samples/cash_application_engine.py` and compare `python3 -m harness eval` with the previous run.
- The `offline` provider must keep producing valid documents. It is the CI path and the demo fallback.
- Keep the server bound to 127.0.0.1, with its Host check and `X-Harness` header check.
