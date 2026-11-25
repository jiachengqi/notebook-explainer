"""Configuration: harness.toml, then .env, then environment variables, then command-line flags.

The provider is switched with ``provider = "..."`` in harness.toml, with the
HARNESS_PROVIDER environment variable, with ``--provider`` or from the UI.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "provider": "gemini",
    "server": {"host": "127.0.0.1", "port": 8765, "open_browser": True},
    "generation": {
        "temperature": 0.0, "timeout_seconds": 180, "max_repair_rounds": 2, "on_unverified": "flag",
        "parallel": True, "cache": True, "coverage_audiences": ["agent"],
    },
    "paths": {"runs_dir": "runs", "cache_dir": ".cache/llm"},
    "policy": {"local_only": False},
    "providers": {
        "gemini": {"kind": "openai_compat", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "model": "gemini-3.8-flash", "api_key_env": "GEMINI_API_KEY", "response_format": "json_schema",
                   "temperature": 1.0, "reasoning_effort": "low", "timeout_seconds": 600},
        "offline": {"kind": "offline", "model": "static-analysis"},
    },
}


@dataclass
class Config:
    provider: str
    providers: dict[str, dict]
    server: dict
    generation: dict
    paths: dict
    policy: dict = field(default_factory=dict)
    source: str = "defaults"
    notes: list[str] = field(default_factory=list)

    def provider_cfg(self, name: str | None = None, model: str | None = None) -> tuple[str, dict]:
        name = name or self.provider
        if name not in self.providers:
            raise KeyError(f"unknown provider '{name}'. Configured: {', '.join(self.providers)}")
        cfg = {**self.providers[name]}
        cfg.setdefault("temperature", self.generation.get("temperature", 0.0))
        cfg.setdefault("timeout_seconds", self.generation.get("timeout_seconds", 180))
        if model:
            cfg["model"] = model
        return name, cfg

    def path(self, key: str) -> Path:
        p = Path(self.paths[key])
        return p if p.is_absolute() else ROOT / p


def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    loaded = []
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _merge(base: dict, override: dict) -> dict:
    out = {**base}
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None = None) -> Config:
    candidates = [Path(path)] if path else [Path.cwd() / "harness.toml", ROOT / "harness.toml"]
    data, source = DEFAULTS, "defaults"
    for c in candidates:
        if c.is_file():
            with c.open("rb") as fh:
                data = _merge(DEFAULTS, tomllib.load(fh))
            source = str(c)
            break
    notes = []
    for env_file in {Path.cwd() / ".env", ROOT / ".env"}:
        keys = load_dotenv(env_file)
        if keys:
            notes.append(f"loaded {', '.join(keys)} from {env_file}")
    provider = os.environ.get("HARNESS_PROVIDER") or data["provider"]
    server = {**data["server"]}
    if os.environ.get("HARNESS_PORT"):
        server["port"] = int(os.environ["HARNESS_PORT"])
    providers = {k: dict(v) for k, v in data["providers"].items()}
    if source != "defaults":  # keep the order of harness.toml (the UI dropdown follows it)
        with Path(source).open("rb") as fh:
            order = list(tomllib.load(fh).get("providers", {}))
        providers = {k: providers[k] for k in order + [k for k in providers if k not in order]}
    if os.environ.get("HARNESS_MODEL") and provider in providers:
        providers[provider]["model"] = os.environ["HARNESS_MODEL"]
    policy = {**data.get("policy", {})}
    if os.environ.get("HARNESS_LOCAL_ONLY"):
        policy["local_only"] = os.environ["HARNESS_LOCAL_ONLY"].strip().lower() in {"1", "true", "yes", "on"}
    return Config(provider, providers, server, data["generation"], data["paths"], policy, source, notes)
