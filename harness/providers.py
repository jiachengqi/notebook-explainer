"""LLM providers behind one small interface.

Switch provider in ``harness.toml`` (``provider = "..."``), with the
``HARNESS_PROVIDER`` environment variable, with ``--provider`` on the command
line, or from the dropdown in the web UI.

Three adapter kinds cover most models:

* ``openai_compat``: any endpoint that speaks the OpenAI chat-completions
  protocol. That includes Google Gemini (through its OpenAI-compatible endpoint),
  a local Ollama or LM Studio model, vLLM, and OpenAI itself.
* ``anthropic``: the Anthropic Messages API.
* ``offline``: no model at all. It writes the documents from static analysis
  only. It is used by the tests and as a fallback in demos.

The adapters use only the Python standard library (``urllib``). No SDK sits in
the path that carries API keys and proprietary code.

Keys are pinned: a well-known key such as ``GEMINI_API_KEY`` is only ever sent to
its own provider's host, and no key is sent over plain HTTP except to localhost.
A ``harness.toml`` in a cloned repository therefore cannot redirect your key.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

KEY_HOSTS = {
    "GEMINI_API_KEY": {"generativelanguage.googleapis.com"},
    "ANTHROPIC_API_KEY": {"api.anthropic.com"},
    "OPENAI_API_KEY": {"api.openai.com"},
}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    notes: list[str] = field(default_factory=list)
    host: str = ""
    request_bytes: int = 0


class ProviderError(RuntimeError):
    """A provider call failed. The message is shown to the user."""


class QuotaError(ProviderError):
    """Rate limit or quota exhausted (HTTP 429)."""

    daily = False   # True when the quota resets in hours, so retrying now is pointless


class Provider:
    kind = "base"

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = dict(cfg)
        self.model = cfg.get("model", "")
        self.timeout = float(cfg.get("timeout_seconds", 180))
        self.temperature = cfg.get("temperature", 0.0)

    def api_key(self) -> str | None:
        env = self.cfg.get("api_key_env") or ""
        return os.environ.get(env) if env else None

    def needs_key(self) -> bool:
        return bool(self.cfg.get("api_key_env"))

    def describe(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "model": self.model,
            "base_url": self.cfg.get("base_url", ""),
            "key_env": self.cfg.get("api_key_env", ""),
            "key_present": (not self.needs_key()) or bool(self.api_key()),
        }

    def complete(self, system: str, user: str, schema: dict | None = None,
                 context: dict | None = None) -> Completion:
        raise NotImplementedError

    @property
    def host(self) -> str:
        return urlparse(self.cfg.get("base_url", "")).hostname or ""

    @property
    def is_local(self) -> bool:
        return self.kind == "offline" or self.host in LOCAL_HOSTS

    def _require_key(self) -> str | None:
        if not self.needs_key():
            return None
        key = self.api_key()
        if not key:
            raise ProviderError(
                f"{self.cfg['api_key_env']} is not set. Add it to .env or export it, "
                f"or switch provider (e.g. HARNESS_PROVIDER=offline)."
            )
        self._check_destination()
        return key

    def _check_destination(self) -> None:
        env = self.cfg.get("api_key_env", "")
        url = urlparse(self.cfg.get("base_url", ""))
        host = url.hostname or ""
        allowed = KEY_HOSTS.get(env)
        if allowed and host not in allowed:
            raise ProviderError(
                f"Refusing to send {env} to {host or 'an unknown host'}: that key is only sent to "
                f"{', '.join(sorted(allowed))}. For another endpoint, use a different key variable."
            )
        if url.scheme != "https" and host not in LOCAL_HOSTS:
            raise ProviderError(f"Refusing to send an API key over plain HTTP to {host}. Use https.")


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")[:3000]
        message = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            err = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            message = err.get("message", raw) if isinstance(err, dict) else str(err)
        except (json.JSONDecodeError, AttributeError):
            pass
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if exc.code == 429:
            raise _quota_error(url, message, raw, retry_after) from None
        err_obj = ProviderError(f"HTTP {exc.code} from {url}: {message}")
        err_obj.status = exc.code  # type: ignore[attr-defined]
        err_obj.retry_after = retry_after  # type: ignore[attr-defined]
        raise err_obj from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from None
    except TimeoutError:
        # Sent, but no answer in time. Connection failures surface as URLError above.
        raise ProviderError(f"{url} did not answer within {timeout:.0f} s "
                            f"(raise timeout_seconds or lower reasoning_effort in harness.toml)") from None


def _size(body: dict) -> int:
    return len(json.dumps(body).encode("utf-8"))


def _quota_error(url: str, message: str, raw: str, retry_after: str | None) -> QuotaError:
    wait = _retry_seconds(raw, retry_after)
    daily = bool(re.search(r"per ?day|perday|daily", raw, re.I)) or (wait is not None and wait > 300)
    if daily:
        when = f" It resets in about {_human(wait)}." if wait else ""
        err = QuotaError(f"The provider's quota is used up ({urlparse(url).hostname}).{when} "
                         "Switch provider, enable billing, or try again later.")
    else:
        err = QuotaError(f"HTTP 429 from {url}: {message}")
    err.status = 429  # type: ignore[attr-defined]
    err.retry_after = retry_after  # type: ignore[attr-defined]
    err.daily = daily
    return err


def _retry_seconds(raw: str, retry_after: str | None) -> float | None:
    if retry_after and str(retry_after).isdigit():
        return float(retry_after)
    m = re.search(r"retry in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", raw)
    if m and m.group(1):
        h = re.search(r"(\d+)h", m.group(1))
        mi = re.search(r"(\d+)m", m.group(1))
        sec = re.search(r"([\d.]+)s", m.group(1))
        return (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + \
            (float(sec.group(1)) if sec else 0)
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', raw)
    return float(m.group(1)) if m else None


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def _with_retries(call, retries: int = 2):
    """Retry rate limits and server errors, with longer waits when the provider is overloaded."""
    for attempt in range(retries + 1):
        try:
            return call()
        except ProviderError as exc:
            status = getattr(exc, "status", None)
            retryable = (status == 429 and not getattr(exc, "daily", False)) or (status is not None and status >= 500)
            if not retryable or attempt == retries:
                raise
            wait = (5, 15)[attempt] if status == 503 else 2 ** attempt * 2
            ra = getattr(exc, "retry_after", None)
            if ra and str(ra).isdigit():
                wait = min(int(ra), 30)
            time.sleep(wait)
    raise AssertionError("unreachable")


class OpenAICompatProvider(Provider):
    kind = "openai_compat"

    def complete(self, system, user, schema=None, context=None) -> Completion:
        key = self._require_key()
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        mode = self.cfg.get("response_format", "json_schema")
        if mode == "json_schema" and schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "notebook_document", "schema": schema}}
        elif mode in {"json_object", "json_schema"}:
            body["response_format"] = {"type": "json_object"}
        if self.cfg.get("max_tokens"):
            body["max_tokens"] = int(self.cfg["max_tokens"])
        if self.cfg.get("reasoning_effort"):
            body["reasoning_effort"] = str(self.cfg["reasoning_effort"])

        notes: list[str] = []
        for _ in range(4):
            try:
                data = _with_retries(lambda: _post_json(url, headers, body, self.timeout))
                break
            except ProviderError as exc:
                status = getattr(exc, "status", None)
                msg = str(exc).lower()
                if status not in (400, 422):
                    raise
                # Some endpoints reject response_format, temperature or reasoning_effort: drop the one named.
                if "response_format" in body and ("response_format" in msg or "schema" in msg or "json" in msg):
                    body.pop("response_format")
                    notes.append("the provider rejected response_format; retried without it")
                elif "temperature" in body and "temperature" in msg:
                    body.pop("temperature")
                    notes.append("the provider rejected temperature; retried with its default")
                elif "reasoning_effort" in body and "reasoning" in msg:
                    body.pop("reasoning_effort")
                    notes.append("the provider rejected reasoning_effort; retried with its default")
                elif "response_format" in body:
                    body.pop("response_format")
                    notes.append("HTTP 400; retried without response_format")
                else:
                    raise
        else:
            raise ProviderError("the provider kept rejecting the request")

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"unexpected response shape: {str(data)[:300]}") from None
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            refusal = message.get("refusal")
            raise ProviderError(f"empty answer from the model{': ' + refusal if refusal else ''}")
        usage = data.get("usage") or {}
        finish = data["choices"][0].get("finish_reason")
        if finish == "length":
            notes.append("the answer was cut off at the token limit")
        return Completion(content, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                          data.get("model", self.model), notes, self.host, _size(body))


class AnthropicProvider(Provider):
    kind = "anthropic"

    def complete(self, system, user, schema=None, context=None) -> Completion:
        key = self._require_key()
        url = self.cfg.get("base_url", "https://api.anthropic.com/v1").rstrip("/") + "/messages"
        headers = {"x-api-key": key or "", "anthropic-version": "2023-06-01"}
        body: dict = {
            "model": self.model,
            "max_tokens": int(self.cfg.get("max_tokens", 16000)),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        notes: list[str] = []
        try:
            data = _with_retries(lambda: _post_json(url, headers, body, self.timeout))
        except ProviderError as exc:
            if getattr(exc, "status", None) == 400 and "temperature" in str(exc).lower():
                body.pop("temperature")
                notes.append("the provider rejected temperature; retried with its default")
                data = _with_retries(lambda: _post_json(url, headers, body, self.timeout))
            else:
                raise
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text:
            raise ProviderError(f"empty answer from the model: {str(data)[:300]}")
        if data.get("stop_reason") == "max_tokens":
            notes.append("the answer was cut off at max_tokens")
        usage = data.get("usage") or {}
        return Completion(text, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
                          data.get("model", self.model), notes, self.host, _size(body))


class OfflineProvider(Provider):
    """No model: documents are written from static analysis only."""
    kind = "offline"

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.model = self.model or "static-analysis"

    def complete(self, system, user, schema=None, context=None) -> Completion:
        from .offline import offline_answer
        if not context:
            raise ProviderError("the offline provider needs the analysis context")
        return Completion(json.dumps(offline_answer(context)), 0, 0, "static-analysis")


KINDS = {"openai_compat": OpenAICompatProvider, "anthropic": AnthropicProvider, "offline": OfflineProvider}


def make_provider(name: str, cfg: dict) -> Provider:
    kind = cfg.get("kind", "openai_compat")
    if kind not in KINDS:
        raise ProviderError(f"unknown provider kind '{kind}' for '{name}' (use one of {', '.join(KINDS)})")
    return KINDS[kind](name, cfg)
