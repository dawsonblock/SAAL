"""Stable, intentionally restricted serialization for commitments and audit hashes."""
import hashlib
import hmac
import json
import math
from typing import Any


def _guard(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in security-critical data")
        return
    if isinstance(value, list):
        for item in value:
            _guard(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("security-critical keys must be strings")
        for item in value.values():
            _guard(item)
        return
    raise TypeError(f"unsupported JSON value type: {type(value)!r}")


def encode(value: Any) -> bytes:
    _guard(value)
    # Narrow JSON subset. If using across languages, replace with RFC 8785 JCS.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def sign(value: Any, key: bytes) -> str:
    if len(key) < 32:
        raise ValueError("signing key must have at least 32 bytes")
    return hmac.new(key, encode(value), hashlib.sha256).hexdigest()


def verify(value: Any, signature: str, key: bytes) -> bool:
    if not isinstance(signature, str) or len(signature) != 64:
        return False
    return hmac.compare_digest(sign(value, key), signature)
