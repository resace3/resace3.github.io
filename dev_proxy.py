from __future__ import annotations

import http.server
import mimetypes
import os
import posixpath
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SLEEP_APP = "http://127.0.0.1:5051"
STATIC_SNAPSHOT_PREFIX = "/__static/"
SLEEP_PREFIXES = (
    "/static/",
    "/dag/",
    "/entities/",
    "/saved-analyses",
    "/sensor-maps",
    "/settings",
    "/analyses/",
)
SLEEP_EXACT = {
    "/causal-dag.html",
    "/run",
    "/stepwise",
    "/outcome-histogram",
    "/covariate-histogram",
}


class DevProxyHandler(http.server.SimpleHTTPRequestHandler):
    server_version = "PersonalWebsiteDevProxy/1.0"

    def translate_path(self, path: str) -> str:
        path = urllib.parse.urlsplit(path).path
        if path.startswith(STATIC_SNAPSHOT_PREFIX):
            path = "/" + path[len(STATIC_SNAPSHOT_PREFIX):]
        path = posixpath.normpath(urllib.parse.unquote(path))
        parts = [part for part in path.split("/") if part and part not in (os.curdir, os.pardir)]
        local_path = ROOT
        for part in parts:
            local_path /= part
        return str(local_path)

    def do_GET(self) -> None:
        if self._should_proxy_to_sleep():
            self._proxy_to_sleep()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._should_proxy_to_sleep():
            self._proxy_to_sleep()
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        if self._should_proxy_to_sleep():
            self._proxy_to_sleep()
            return
        self.send_error(404, "Not found")

    def _should_proxy_to_sleep(self) -> bool:
        path = urllib.parse.urlsplit(self.path).path
        return path in SLEEP_EXACT or any(path.startswith(prefix) for prefix in SLEEP_PREFIXES)

    def _sleep_target_url(self) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        path = "/" if parsed.path == "/causal-dag.html" else parsed.path
        base = urllib.parse.urlsplit(SLEEP_APP)
        return urllib.parse.urlunsplit((base.scheme, base.netloc, path, parsed.query, ""))

    def _proxy_to_sleep(self) -> None:
        body = None
        if self.command not in {"GET", "HEAD"}:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        request = urllib.request.Request(
            self._sleep_target_url(),
            data=body,
            headers=headers,
            method=self.command,
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    payload = self._rewrite_sleep_html(payload)
                self.send_response(response.status)
                self._send_proxy_headers(response.headers, len(payload))
                if self.command != "HEAD":
                    self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self._send_proxy_headers(error.headers, len(payload))
            if self.command != "HEAD":
                self.wfile.write(payload)
        except OSError as error:
            self.send_error(502, f"Sleep_Causal app is unavailable: {error}")

    def _send_proxy_headers(self, headers, content_length: int) -> None:
        skipped = {"connection", "content-length", "transfer-encoding", "content-encoding"}
        for key, value in headers.items():
            if key.lower() not in skipped:
                if key.lower() == "location":
                    value = self._rewrite_sleep_location(value)
                self.send_header(key, value)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _rewrite_sleep_html(self, payload: bytes) -> bytes:
        html = payload.decode("utf-8", errors="replace")
        html = html.replace('href="/"', 'href="/causal-dag.html"')
        html = html.replace("href='/'", "href='/causal-dag.html'")
        return html.encode("utf-8")

    def _rewrite_sleep_location(self, location: str) -> str:
        parsed = urllib.parse.urlsplit(location)
        if parsed.path == "/" and parsed.netloc in {"", "127.0.0.1:5051", "127.0.0.1:5050"}:
            return "/causal-dag.html"
        return location


if __name__ == "__main__":
    mimetypes.add_type("text/javascript", ".js")
    http.server.ThreadingHTTPServer(("0.0.0.0", 8000), DevProxyHandler).serve_forever()
