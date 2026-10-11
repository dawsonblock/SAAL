"""Reproducible content digest for all SAAL Python implementation modules.

This is a self-consistency checksum, not a signed provenance attestation.
An independently controlled qualifier must use its own trusted source copy.
"""
import hashlib
from pathlib import Path
from .canonical import digest


def implementation_digest() -> str:
    source_root = Path(__file__).resolve().parent
    files = sorted(source_root.glob("*.py"))
    if not files:
        raise ValueError("missing implementation source files")
    return digest({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
