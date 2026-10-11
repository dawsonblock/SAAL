"""Verify the shipped content inventory against MANIFEST.sha256.

Run from any directory: python scripts/verify_release.py
The manifest is a self-consistency checksum, not an authenticated signature.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'MANIFEST.sha256'


def main() -> int:
    errors = []
    seen = set()
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        value, sep, rel = line.partition('  ')
        if (not sep or len(value) != 64 or
                any(c not in '0123456789abcdef' for c in value) or
                not rel or rel.startswith('/') or '..' in Path(rel).parts or
                rel in seen):
            errors.append('invalid manifest entry')
            continue
        seen.add(rel)
        file = ROOT / rel
        if file.is_symlink() or not file.is_file():
            errors.append(f'missing or symlink: {rel}')
            continue
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != value:
            errors.append(f'mismatch: {rel}')
    if not seen:
        errors.append('empty manifest')
    available = set()
    for f in ROOT.rglob('*'):
        if (f.is_file() or f.is_symlink()) and f != MANIFEST and            '__pycache__' not in f.parts and f.suffix not in {'.pyc', '.pyo'} and            '.pytest_cache' not in f.parts and '.egg-info' not in str(f):
            available.add(f.relative_to(ROOT).as_posix())
    if available != seen:
        errors.append(f'manifest inventory incomplete: missing={sorted(available-seen)} '
                      f'extra={sorted(seen-available)}')
    if errors:
        for problem in errors:
            print(problem, file=sys.stderr)
        return 1
    print(f'PASS: {len(seen)} manifest file hashes match (not an authenticity signature)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
