"""Verify every file and reject missing, extra, unsafe or duplicated manifest paths."""
from pathlib import Path
import hashlib
import sys


def verify(root):
    root=Path(root).resolve();expected={}
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        target=(root/name).resolve()
        if name in expected or not target.is_relative_to(root) or len(digest)!=64:
            raise ValueError('unsafe or duplicate manifest entry')
        expected[name]=digest
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)
            and p.suffix not in {'.pyc','.pyo'} and p.name!='MANIFEST.sha256'}
    if set(expected)!=actual:raise ValueError(f'inventory mismatch: missing={set(expected)-actual}, extra={actual-set(expected)}')
    for name,digest in expected.items():
        p=root/name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('hash mismatch or symlink: '+name)
    return len(expected)

if __name__=='__main__':print('Verified files:',verify(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parents[1]))
