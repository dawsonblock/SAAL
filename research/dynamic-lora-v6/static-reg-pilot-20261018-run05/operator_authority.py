"""One-campaign, single-operator receipt service for a development-only run.

This is intentionally not represented as independent custody. It supports only
five predeclared attempts, keeps signing keys on the Mac, and exposes narrowly
scoped calls through a short-lived authenticated tunnel.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import stat
import tempfile
import time
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from saal_rsi.model_contract import sign, verify
from saal_rsi.spec import canonical, parse_unique_json, sha

MAX_REQUEST = 110_000_000
MAX_ARTIFACT = 100_000_000
ZERO = "sha256:" + "0" * 64


def read_secret(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("private authority input must be a regular file")
    info = path.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("private authority input permissions are not private")
    return path.read_bytes()


class Authority:
    def __init__(self, config_path: Path, key_path: Path, ledger_path: Path,
                 artifact_root: Path):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.config = parse_unique_json(read_secret(config_path))
        self.key = Ed25519PrivateKey.from_private_bytes(read_secret(key_path))
        self.pin = self.key.public_key().public_bytes_raw().hex()
        if ledger_path.exists() or ledger_path.is_symlink():
            raise FileExistsError("fresh one-campaign ledger path required")
        self.ledger_path = ledger_path
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.states = {}
        self.tokens = {}
        for item in self.config["attempts"]:
            attempt = item["attempt_id"]
            if attempt in self.states:
                raise ValueError("duplicate attempt identity")
            self.states[attempt] = {"expected": item["bindings"], "admission": item["admission"],
                                    "status": "RESERVED", "receipt": None,
                                    "terminal": None, "artifact": None}
            self.tokens[attempt] = item["token"]
        if not 1 <= len(self.states) <= 5:
            raise ValueError("one bounded pilot campaign required")
        self.sequence = 0
        self.previous = ZERO

    def append_event(self, attempt: str, kind: str, details: dict) -> None:
        self.sequence += 1
        body = {
            "schema": "saal-static-pilot-local-ledger-event-v1",
            "campaign_id": self.config["campaign_id"],
            "sequence": self.sequence,
            "previous": self.previous,
            "attempt_id": attempt,
            "kind": kind,
            "details_digest": sha(details),
            "time": int(time.time()),
        }
        envelope = sign(body, self.key)
        raw = canonical(envelope) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.ledger_path, flags, 0o600)
        try:
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        self.previous = sha(envelope)

    def consume(self, attempt: str, request: dict) -> dict:
        state = self.states[attempt]
        if set(request) != {"bindings", "challenge"} or state["status"] != "RESERVED":
            raise PermissionError("attempt unavailable")
        challenge = request["challenge"]
        if type(challenge) is not str or not re.fullmatch(r"[0-9a-f]{64}", challenge):
            raise ValueError("invalid one-use challenge")
        if canonical(request["bindings"]) != canonical(state["expected"]):
            raise PermissionError("attempt binding mismatch")
        now = int(time.time())
        admission = state["admission"]
        if now < admission["issued_at"] or now >= admission["expires_at"]:
            raise PermissionError("admission outside validity window")
        payload = {
            "schema": "saal-static-pilot-attempt-receipt-v1",
            "campaign_id": admission["campaign_id"],
            "attempt_id": attempt,
            "admission_digest": state["expected"]["admission_digest"],
            "request_nonce": challenge,
            "configuration_digest": admission["configuration_digest"],
            "dataset_digest": admission["dataset_digest"],
            "bindings_digest": sha(state["expected"]),
            "sequence": self.sequence + 1,
            "status": "CONSUMED",
            "consumed_at": now,
        }
        receipt = sign(payload, self.key)
        self.append_event(attempt, "CONSUMED", {"receipt_digest": sha(receipt),
                                                  "bindings_digest": sha(state["expected"])})
        state["status"] = "CONSUMED"
        state["receipt"] = receipt
        return receipt

    def finalize(self, attempt: str, request: dict) -> dict:
        state = self.states[attempt]
        if set(request) != {"attempt_receipt", "outcome", "outcome_digest", "finish_nonce"}:
            raise ValueError("terminal request shape")
        if state["status"] != "CONSUMED" or canonical(request["attempt_receipt"]) != canonical(state["receipt"]):
            raise PermissionError("attempt receipt absent, substituted or already finalized")
        if request["outcome"] not in ("COMPLETED", "FAILED"):
            raise ValueError("invalid terminal disposition")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", request["outcome_digest"]):
            raise ValueError("invalid outcome digest")
        nonce = request["finish_nonce"]
        if type(nonce) is not str or not re.fullmatch(r"[0-9a-f]{64}", nonce):
            raise ValueError("invalid terminal challenge")
        consumed = verify(state["receipt"], self.pin)
        now = int(time.time())
        body = {
            "schema": "saal-static-pilot-outcome-receipt-v1",
            "campaign_id": self.config["campaign_id"],
            "attempt_id": attempt,
            "admission_digest": state["expected"]["admission_digest"],
            "attempt_receipt_digest": sha(state["receipt"]),
            "outcome_digest": request["outcome_digest"],
            "request_nonce": nonce,
            "sequence": self.sequence + 1,
            "status": request["outcome"],
            "finalized_at": now,
        }
        receipt = sign(body, self.key)
        self.append_event(attempt, request["outcome"], {
            "attempt_receipt_digest": sha(state["receipt"]),
            "outcome_digest": request["outcome_digest"],
            "consumed_sequence": consumed["sequence"],
        })
        state["status"] = request["outcome"]
        state["terminal"] = receipt
        return receipt

    def accept_artifact(self, attempt: str, raw: bytes) -> dict:
        state = self.states[attempt]
        if state["status"] not in ("COMPLETED", "FAILED") or state["artifact"] is not None:
            raise PermissionError("artifact upload is not authorized")
        if not 1 <= len(raw) <= MAX_ARTIFACT:
            raise ValueError("artifact size bound")
        from io import BytesIO
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (not infos or len(names) != len(set(names)) or archive.testzip() is not None
                    or sum(item.file_size for item in infos) > MAX_ARTIFACT
                    or any(item.is_dir() or item.filename.startswith("/")
                           or ".." in Path(item.filename).parts
                           or ((item.external_attr >> 16) & 0o170000) == 0o120000
                           for item in infos)):
                raise ValueError("artifact archive rejected")
        digest = hashlib.sha256(raw).hexdigest()
        target = self.artifact_root / f"{attempt}.zip"
        fd, temporary = tempfile.mkstemp(prefix=f".{attempt}-", suffix=".partial",
                                         dir=self.artifact_root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
            os.unlink(temporary)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        receipt = {"attempt_id": attempt, "size": len(raw), "sha256": digest,
                   "members": sorted(names), "status": "RECEIVED"}
        self.append_event(attempt, "ARTIFACT_RECEIVED", receipt)
        state["artifact"] = receipt
        return receipt


class Handler(BaseHTTPRequestHandler):
    server_version = "SAAL-Pilot/1"
    sys_version = ""

    def log_message(self, *_args):
        return

    def respond(self, status: int, body: dict) -> None:
        raw = canonical(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/healthz":
            self.respond(200, {"status": "ready"})
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        try:
            match = re.fullmatch(r"/v1/(consume|finalize|artifact)/([A-Za-z0-9_-]{1,80})", self.path)
            if match is None:
                raise PermissionError("route unavailable")
            operation, attempt = match.groups()
            if attempt not in self.server.authority.states:
                raise PermissionError("attempt unavailable")
            expected = self.server.authority.tokens[attempt]
            actual = self.headers.get("Authorization", "")
            if not hmac.compare_digest(actual, "Bearer " + expected):
                raise PermissionError("request denied")
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit() or int(length) > MAX_REQUEST:
                raise ValueError("request size bound")
            raw = self.rfile.read(int(length))
            if len(raw) != int(length):
                raise ValueError("incomplete request")
            authority = self.server.authority
            if operation == "artifact":
                result = authority.accept_artifact(attempt, raw)
            else:
                request = parse_unique_json(raw)
                result = (authority.consume(attempt, request) if operation == "consume"
                          else authority.finalize(attempt, request))
            self.respond(200, {"result": result})
        except PermissionError:
            self.respond(403, {"error": "request denied"})
        except (ValueError, OSError, zipfile.BadZipFile):
            self.respond(400, {"error": "request rejected"})
        except Exception:  # noqa: BLE001 - never leak exception or request details remotely
            self.respond(500, {"error": "authority error"})


class Server(HTTPServer):
    def __init__(self, address, authority):
        self.authority = authority
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    authority = Authority(args.config, args.key, args.ledger, args.artifacts)
    server = Server(("127.0.0.1", args.port), authority)
    server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
