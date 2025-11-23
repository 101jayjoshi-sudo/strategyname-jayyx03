#!/usr/bin/env python3
"""HTTP endpoints exposing health, runtime settings, and control hooks."""

from __future__ import annotations

import json
import time
import hmac
import hashlib
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from strategy_interface import available_strategies


MAX_SKEW_MS = 5 * 60 * 1000


class BotHTTPServer:
    """Lightweight HTTP server that surfaces bot status and settings."""

    def __init__(self, bot, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.bot = bot
        self.host = host
        self.port = port
        self._server = ThreadingHTTPServer((self.host, self.port), self._handler_factory())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler_factory(self):
        bot = self.bot

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: HTTPStatus, payload: Any) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args):  # noqa: D401 - silence default logging
                return

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    payload = bot.get_status()
                    payload["strategies"] = available_strategies()
                    self._send_json(HTTPStatus.OK, payload)
                    return
                if parsed.path == "/settings":
                    self._send_json(HTTPStatus.OK, bot.get_settings())
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

            def do_POST(self):  # noqa: N802 - not supported on this port
                self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "POST not supported"})

        return Handler

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class BotControlServer:
    """Control plane server exposing settings, performance, and command hooks."""

    def __init__(
        self,
        bot,
        *,
        host: str = "0.0.0.0",
        port: int = 3010,
        bot_secret: Optional[str] = None,
    ) -> None:
        self.bot = bot
        self.host = host
        self.port = port
        self.bot_secret = bot_secret
        self._server = ThreadingHTTPServer((self.host, self.port), self._handler_factory())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler_factory(self):
        bot = self.bot
        bot_secret = self.bot_secret

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args):  # noqa: D401
                return

            def _send_json(self, status: HTTPStatus, payload: Any) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return b""
                return self.rfile.read(length)

            def _verify_hmac(self, payload: Any, raw_body: bytes) -> Optional[str]:
                if not bot_secret:
                    return "HMAC secret is not configured"

                signature = self.headers.get("X-Bot-Signature")
                timestamp = self.headers.get("X-Bot-Timestamp")
                if not signature or not timestamp:
                    return "Missing authentication headers"

                try:
                    request_time = int(timestamp)
                except ValueError:
                    return "Invalid timestamp"

                now = int(time.time() * 1000)
                if abs(now - request_time) > MAX_SKEW_MS:
                    return "Request timestamp outside allowed window"

                canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                digest_canonical = hmac.new(
                    bot_secret.encode("utf-8"),
                    canonical.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                digest_raw = hmac.new(
                    bot_secret.encode("utf-8"),
                    raw_body,
                    hashlib.sha256,
                ).hexdigest()

                if not hmac.compare_digest(signature, digest_canonical) and not hmac.compare_digest(signature, digest_raw):
                    return "Invalid signature"

                return None

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/settings":
                    self._send_json(HTTPStatus.OK, bot.get_settings())
                    return
                if parsed.path == "/performance":
                    self._send_json(HTTPStatus.OK, bot.get_performance())
                    return
                if parsed.path == "/logs":
                    self._send_json(HTTPStatus.OK, bot.get_logs())
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path not in {"/settings", "/commands"}:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                    return

                raw_body = self._read_body()
                if not raw_body:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing request body"})
                    return

                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                error = self._verify_hmac(payload, raw_body)
                if error:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": error})
                    return

                try:
                    if parsed.path == "/settings":
                        bot.apply_settings(payload)
                        self._send_json(HTTPStatus.OK, bot.get_settings())
                        return

                    if parsed.path == "/commands":
                        command = str(payload.get("command", "")).lower()
                        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                        result = bot.handle_command(command, metadata)
                        self._send_json(HTTPStatus.OK, result)
                        return
                except Exception as exc:  # noqa: BLE001 - surface as JSON error
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return

        return Handler

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
