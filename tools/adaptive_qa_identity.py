"""Bind collected QA source bytes to the commit captured before collection."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile


def source_identity(repo, files, commit):
    repo = Path(repo)
    names = sorted(n for n in files if Path(n).suffix == '.py' or n == 'pyproject.toml'
                   or n.startswith('configs/') and Path(n).suffix in ('.json', '.txt', '.toml', '.yaml', '.yml'))
    if not names:
        raise ValueError('QA implementation source is missing')
    tracked = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines())
    if any(n not in tracked for n in names):
        raise ValueError('uncommitted QA source: '+next(n for n in names if n not in tracked))
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit, '--', *names], cwd=repo)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        for member in tree:
            if not member.isfile():
                continue
            data = tree.extractfile(member).read()
            if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('LFS pointer is not executable QA source')
            hashes[member.name] = hashlib.sha256(data).hexdigest()
    if set(hashes) != set(names):
        raise ValueError('incomplete committed QA source inventory')
    for name, expected in hashes.items():
        with Path(files[name]).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('QA source differs from captured commit: '+name)
    return {'schema': 'saal-qa-source-identity-v1', 'source_commit': commit, 'files': hashes}
