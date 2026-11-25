"""Provider adapters against a local fake HTTP server: request shape, parsing, fallbacks, errors."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness.providers import ProviderError, QuotaError, make_provider


class FakeAPI:
    def __init__(self):
        self.requests = []
        self.responses = []          # list of (status, dict)
        self.delay = 0.0             # seconds to wait before answering, to simulate a slow model
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {k.lower(): v for k, v in self.headers.items()}
                api.requests.append({"path": self.path, "headers": headers, "body": body})
                status, payload = api.responses.pop(0)
                data = json.dumps(payload).encode()
                if api.delay:
                    time.sleep(api.delay)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass             # the client gave up waiting

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def chat_ok(text, model="m"):
    return 200, {"model": model, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class OpenAICompatTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        os.environ["TEST_KEY"] = "secret-123"
        self.p = make_provider("gemini", {"kind": "openai_compat", "base_url": self.api.url + "/v1beta/openai/",
                                          "model": "gemini-x", "api_key_env": "TEST_KEY",
                                          "response_format": "json_schema", "temperature": 0.0})

    def tearDown(self):
        self.api.close()

    def test_request_shape_and_parse(self):
        self.api.responses.append(chat_ok('{"ok": true}'))
        comp = self.p.complete("sys", "user", {"type": "object"})
        req = self.api.requests[0]
        self.assertEqual(req["path"], "/v1beta/openai/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer secret-123")
        self.assertEqual(req["body"]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(req["body"]["response_format"]["type"], "json_schema")
        self.assertEqual(req["body"]["temperature"], 0.0)
        self.assertNotIn("reasoning_effort", req["body"])
        self.assertEqual(comp.text, '{"ok": true}')
        self.assertEqual((comp.input_tokens, comp.output_tokens), (11, 7))

    def test_sends_reasoning_effort_when_configured(self):
        p = make_provider("gemini", {**self.p.cfg, "reasoning_effort": "low"})
        self.api.responses.append(chat_ok("{}"))
        p.complete("s", "u", None)
        self.assertEqual(self.api.requests[0]["body"]["reasoning_effort"], "low")

    def test_falls_back_when_reasoning_effort_rejected(self):
        p = make_provider("gemini", {**self.p.cfg, "reasoning_effort": "minimal"})
        self.api.responses.append((400, {"error": {"message": "Invalid reasoning_effort: minimal. "
                                                              "Valid values are: high, low, medium, none"}}))
        self.api.responses.append(chat_ok("{}"))
        comp = p.complete("s", "u", {"type": "object"})
        self.assertNotIn("reasoning_effort", self.api.requests[1]["body"])
        self.assertIn("response_format", self.api.requests[1]["body"])   # only the rejected field is dropped
        self.assertTrue(any("reasoning_effort" in n for n in comp.notes))

    def test_slow_model_times_out_with_a_hint(self):
        p = make_provider("gemini", {**self.p.cfg, "timeout_seconds": 0.3})
        self.api.delay = 1.0
        self.api.responses.append(chat_ok("{}"))
        with self.assertRaises(ProviderError) as ctx:
            p.complete("s", "u", None)
        self.assertIn("did not answer within", str(ctx.exception))   # sent, but no answer: not a network failure
        self.assertIn("timeout_seconds", str(ctx.exception))

    def test_falls_back_when_schema_rejected(self):
        self.api.responses.append((400, {"error": {"message": "Invalid response_format json_schema"}}))
        self.api.responses.append(chat_ok("{}"))
        comp = self.p.complete("s", "u", {"type": "object"})
        self.assertNotIn("response_format", self.api.requests[1]["body"])
        self.assertTrue(any("response_format" in n for n in comp.notes))

    def test_falls_back_when_temperature_rejected(self):
        self.api.responses.append((400, {"error": {"message": "Unsupported value: 'temperature'"}}))
        self.api.responses.append(chat_ok("{}"))
        self.p.complete("s", "u", None)
        self.assertNotIn("temperature", self.api.requests[1]["body"])

    def test_quota_error_after_retries(self):
        for _ in range(3):
            self.api.responses.append((429, {"error": {"message": "quota exceeded"}}))
        import harness.providers as prov
        original = prov.time.sleep
        prov.time.sleep = lambda s: None
        try:
            with self.assertRaises(QuotaError):
                self.p.complete("s", "u", None)
        finally:
            prov.time.sleep = original
        self.assertEqual(len(self.api.requests), 3)

    def test_daily_quota_is_not_retried_and_says_when_it_resets(self):
        self.api.responses.append((429, {"error": {"message": "Quota exceeded for metric: generate_content_free_tier"
                                                              "_requests, limit: 20 requests per day. Please retry in "
                                                              "7h12m43.9s."}}))
        with self.assertRaises(QuotaError) as ctx:
            self.p.complete("s", "u", None)
        self.assertEqual(len(self.api.requests), 1)
        self.assertIn("7h 12m", str(ctx.exception))

    def test_per_minute_limit_waits_as_long_as_the_provider_asks(self):
        import harness.providers as prov
        waits = []
        original = prov.time.sleep
        prov.time.sleep = waits.append
        try:
            self.api.responses += [(429, {"error": {"message": "Quota exceeded for metric: generate_content_free_tier"
                                                               "_requests, limit: 5, description: Generate content "
                                                               "requests per minute limit exceeded on Free Tier. "
                                                               "Please retry in 8.75064651s."}}), chat_ok("{}")]
            self.p.complete("s", "u", None)
        finally:
            prov.time.sleep = original
        self.assertEqual(len(self.api.requests), 2)
        self.assertEqual(len(waits), 1)
        self.assertTrue(9 <= waits[0] <= 11, waits)

    def test_overloaded_provider_gets_longer_waits(self):
        import harness.providers as prov
        waits = []
        original = prov.time.sleep
        prov.time.sleep = waits.append
        try:
            self.api.responses += [(503, {"error": {"message": "high demand"}})] * 2 + [chat_ok("{}")]
            self.p.complete("s", "u", None)
        finally:
            prov.time.sleep = original
        self.assertEqual(waits, [5, 15])

    def test_records_where_the_request_went_and_its_size(self):
        self.api.responses.append(chat_ok("{}"))
        comp = self.p.complete("s", "u", None)
        self.assertEqual(comp.host, "127.0.0.1")
        self.assertGreater(comp.request_bytes, 0)

    def test_missing_key_message(self):
        del os.environ["TEST_KEY"]
        with self.assertRaises(ProviderError) as ctx:
            self.p.complete("s", "u", None)
        self.assertIn("TEST_KEY is not set", str(ctx.exception))

    def test_no_key_needed_for_local_models(self):
        p = make_provider("ollama", {"kind": "openai_compat", "base_url": self.api.url + "/v1", "model": "local",
                                     "api_key_env": ""})
        self.api.responses.append(chat_ok("{}"))
        p.complete("s", "u", None)
        self.assertNotIn("authorization", self.api.requests[0]["headers"])


class KeyPinningTests(unittest.TestCase):
    def test_known_key_only_goes_to_its_own_host(self):
        os.environ["GEMINI_API_KEY"] = "real-key"
        try:
            p = make_provider("gemini", {"kind": "openai_compat", "base_url": "https://evil.example/v1",
                                         "model": "m", "api_key_env": "GEMINI_API_KEY"})
            with self.assertRaises(ProviderError) as ctx:
                p.complete("s", "u", None)
            self.assertIn("Refusing to send GEMINI_API_KEY", str(ctx.exception))
        finally:
            del os.environ["GEMINI_API_KEY"]

    def test_no_key_over_plain_http_to_a_remote_host(self):
        os.environ["TEST_KEY"] = "k"
        p = make_provider("x", {"kind": "openai_compat", "base_url": "http://api.example.com/v1", "model": "m",
                                "api_key_env": "TEST_KEY"})
        with self.assertRaises(ProviderError) as ctx:
            p.complete("s", "u", None)
        self.assertIn("plain HTTP", str(ctx.exception))

    def test_local_and_offline_providers_count_as_local(self):
        self.assertTrue(make_provider("ollama", {"kind": "openai_compat", "base_url": "http://localhost:11434/v1",
                                                 "api_key_env": ""}).is_local)
        self.assertTrue(make_provider("offline", {"kind": "offline"}).is_local)
        self.assertFalse(make_provider("gemini", {"kind": "openai_compat",
                                                  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}
                                       ).is_local)


class AnthropicTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        os.environ["TEST_ANTHROPIC"] = "sk-ant-test"
        self.p = make_provider("anthropic", {"kind": "anthropic", "base_url": self.api.url + "/v1",
                                             "model": "claude-x", "api_key_env": "TEST_ANTHROPIC",
                                             "max_tokens": 1000})

    def tearDown(self):
        self.api.close()

    def test_request_shape_and_parse(self):
        self.api.responses.append((200, {"model": "claude-x", "content": [{"type": "text", "text": '{"a": 1}'}],
                                         "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}}))
        comp = self.p.complete("sys", "user", {"type": "object"})
        req = self.api.requests[0]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(req["body"]["system"], "sys")
        self.assertEqual(req["body"]["max_tokens"], 1000)
        self.assertEqual(comp.text, '{"a": 1}')
        self.assertEqual(comp.output_tokens, 3)


if __name__ == "__main__":
    unittest.main()
