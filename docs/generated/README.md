# Generated documents for `samples/cash_application_engine.py`

These are the submitted documents. To regenerate them with the current code (3 to 12 model calls):

```bash
python3 -m harness generate samples/cash_application_engine.py --out docs/generated
python3 -m harness eval docs/generated --readme README.md
```

- `manager.md`, `developer.md`, `agent.md`: the three documents.
- `report.json`: settings, timings, tokens, what was sent where, prompt hashes, redacted and flagged lines.
- `docs.json`: every claim with its status and the reason when a check failed.
- `eval.json`: the golden-set score.
- `verify.json`: present when the folder was re-checked with `harness verify`.
