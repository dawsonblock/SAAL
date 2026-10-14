import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('qa_identity_tested', ROOT/'tools/adaptive_qa_identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def git(repo, *args):
    return subprocess.check_output(['git', '-c', 'user.name=QA Fixture', '-c',
        'user.email=qa@example.invalid', '-c', 'commit.gpgsign=false', *args], cwd=repo, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path/'src').mkdir()
    source = tmp_path/'src/implementation.py'
    source.write_text('value = 1\n')
    git(tmp_path, 'add', 'src/implementation.py')
    git(tmp_path, 'commit', '-qm', 'first source')
    first = git(tmp_path, 'rev-parse', 'HEAD')
    return tmp_path, source, first


def test_later_head_cannot_relabel_earlier_collected_source(repository, tmp_path):
    repo, source, first = repository
    frozen = tmp_path/'frozen.py'; frozen.write_bytes(source.read_bytes())
    source.write_text('value = 2\n')
    git(repo, 'add', 'src/implementation.py'); git(repo, 'commit', '-qm', 'later source')
    later = git(repo, 'rev-parse', 'HEAD')
    files = {'src/implementation.py': frozen}
    assert identity.source_identity(repo, files, first)['source_commit'] == first
    with pytest.raises(ValueError, match='differs from captured commit'):
        identity.source_identity(repo, files, later)


def test_uncommitted_source_cannot_claim_a_release_commit(repository):
    repo, source, first = repository
    source.write_text('uncommitted = True\n')
    with pytest.raises(ValueError, match='differs from captured commit'):
        identity.source_identity(repo, {'src/implementation.py': source}, first)
