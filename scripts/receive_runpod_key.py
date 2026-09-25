"""One-shot loopback credential intake. No provider calls or browser creation."""
from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import re
import secrets
import stat
import time
from urllib.parse import parse_qs

ROOT = Path.home() / ".config/rsna-knee/temporary-control"


def save_secret(target: Path, value: str) -> None:
    """Never overwrite; never return, log or echo the secret."""
    if not re.fullmatch(r"rpa_[A-Za-z0-9_-]{20,250}", value):
        raise ValueError("Invalid RunPod key format")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def serve(target: Path, *, seconds: int = 120, ready=None) -> bool:
    nonce = secrets.token_urlsafe(32)
    deadline = time.monotonic() + seconds
    saved = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # HTTP request bodies and credentials must never enter logs.

        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; form-action 'self'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body.encode())

        def valid_host(self):
            return self.headers.get("Host") == address

        def do_GET(self):
            if self.path != "/" or not self.valid_host() or time.monotonic() >= deadline:
                return self.reply(403, "Not available")
            self.reply(200, '<!doctype html><title>Private RunPod key intake</title>'
                '<h1>Save temporary RunPod key locally</h1>'
                '<p>No provider request. No key is displayed after saving. This form expires in two minutes.</p>'
                '<form method="post" action="/save">'
                f'<input type="hidden" name="nonce" value="{html.escape(nonce)}">'
                '<label>Temporary key <input type="password" name="key" autocomplete="off" required></label>'
                '<button type="submit">Save privately</button></form>')

        def do_POST(self):
            nonlocal saved
            if (saved or time.monotonic() >= deadline or self.path != "/save"
                or not self.valid_host() or self.headers.get("Origin") != "http://" + address
                or self.headers.get("Content-Type", "").split(";")[0] != "application/x-www-form-urlencoded"):
                return self.reply(403, "Refused")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 1024:
                    raise ValueError()
                self.connection.settimeout(min(2, max(.01, deadline-time.monotonic())))
                values = parse_qs(self.rfile.read(length).decode(), strict_parsing=True)
                if (set(values) != {"nonce", "key"} or any(len(v) != 1 for v in values.values())
                    or not secrets.compare_digest(values["nonce"][0], nonce)
                    or time.monotonic() >= deadline):
                    raise ValueError()
                save_secret(target, values["key"][0].strip())
            except (ValueError, OSError, UnicodeError):
                return self.reply(400, "Not saved. Check format or use a new destination. No key shown.")
            saved = True
            self.reply(200, '<!doctype html><title>Saved privately</title><p>Stored locally; no key shown. Intake is now closed.</p>')

    class BoundedServer(HTTPServer):
        def get_request(self):
            connection, peer = super().get_request()
            connection.settimeout(min(2, max(.01, deadline-time.monotonic())))
            return connection, peer

    with BoundedServer(("127.0.0.1", 0), Handler) as server:
        address = f"127.0.0.1:{server.server_port}"
        server.timeout = .2
        if ready is None:
            print(f"http://{address}/", flush=True)
        else:
            ready(f"http://{address}/")  # Only a loopback URL, never a token or nonce.
        while not saved and time.monotonic() < deadline:
            server.handle_request()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique descriptive filename stem, without secrets")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", args.name):
        parser.error("Use a unique alphanumeric/dash/underscore name")
    for p in [ROOT, *ROOT.parents]:
        if p.is_symlink():
            parser.error("Credential-directory symlinks are refused")
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ROOT.stat().st_uid != os.getuid() or stat.S_IMODE(ROOT.stat().st_mode) != 0o700:
        parser.error("Credential directory must be user-owned mode 0700")
    target = ROOT / (args.name + ".key")
    if target.exists() or target.is_symlink():
        parser.error("Existing destination refused; choose a fresh name")
    ok = serve(target)
    print("SAVED_PRIVATE_FILE" if ok else "EXPIRED_WITHOUT_KEY", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
