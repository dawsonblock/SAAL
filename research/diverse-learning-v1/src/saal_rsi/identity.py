"""Measured local execution identity. No hardware/remote attestation claim."""
from pathlib import Path
import hashlib
import platform
import sys
from importlib.metadata import version
from .spec import sha


def fingerprint():
    root=Path(__file__).resolve().parents[2]
    if (root/'src/saal_rsi/identity.py').resolve()!=Path(__file__).resolve() or not (root/'tests').is_dir() or not (root/'pyproject.toml').is_file():
        raise ValueError('qualification requires the complete extracted source tree (editable install supported)')
    files=[]
    for directory in ('src','tests','configs','tools'):
        base=root/directory
        if base.exists():
            files.extend(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                         and not any(part.endswith('.egg-info') for part in p.parts)
                         and p.suffix not in {'.pyc','.pyo'})
    files.extend(p for p in (root/'pyproject.toml',root/'requirements-lock.txt') if p.exists())
    return sha([(p.relative_to(root).as_posix(),hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(files)])


def runtime_identity():
    return {'implementation_digest':fingerprint(),
            'python_version':sys.version,'python_implementation':platform.python_implementation(),
            'platform':platform.platform(),'machine':platform.machine(),
            'python_executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            'dependencies':{name:version(name) for name in ('cryptography','cffi','pycparser')},
            'limits':'Local measurements, not a trusted host attestation; same-UID administrator can falsify evidence.'}


def require_identity(expected):
    if expected!=runtime_identity(): raise ValueError('runtime/config/source identity mismatch')
