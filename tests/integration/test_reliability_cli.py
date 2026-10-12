import json
import sys
import pytest
from saal_rsi.learning_dataset import semantic_pilot
from saal_rsi.reliability_cli import main, review_key
from saal_rsi.public_review import packet
from saal_rsi.spec import canonical


def test_review_cli_exports_blind_packet_and_does_not_overwrite(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset.json'; dataset.write_bytes(canonical(semantic_pilot()))
    out = tmp_path / 'review'
    monkeypatch.setattr(sys, 'argv', ['reliability', 'review-packet', '--dataset', str(dataset), '--out', str(out)])
    main(); draft = json.loads((out / 'blind-review-draft.json').read_text())
    assert not draft['complete'] and all(r['decision'] == '' for r in draft['records'])
    assert 'Restore a saved draft' in (out / 'blind-review.html').read_text()
    with pytest.raises(FileExistsError): main()


def test_review_cli_cannot_sign_an_incomplete_draft(tmp_path, monkeypatch):
    data = semantic_pilot(); dataset = tmp_path / 'data.json'; dataset.write_bytes(canonical(data))
    draft = tmp_path / 'draft.json'; draft.write_bytes(canonical(packet(data)))
    key = tmp_path / 'key'; key.write_bytes(b'a' * 32); key.chmod(0o600)
    out = tmp_path / 'signed.json'
    monkeypatch.setattr(sys, 'argv', ['reliability', 'sign-review', '--dataset', str(dataset), '--draft', str(draft), '--private-key', str(key), '--out', str(out)])
    with pytest.raises(ValueError, match='incomplete'): main()
    assert not out.exists()


@pytest.mark.parametrize('mutation', ['public', 'symlink', 'oversized', 'hardlink'])
def test_review_key_custody_preflight(tmp_path, mutation):
    key = tmp_path / 'key'; key.write_bytes(b'a' * 32); key.chmod(0o600)
    if mutation == 'public': key.chmod(0o644)
    elif mutation == 'symlink':
        real = tmp_path / 'real'; key.rename(real); key.symlink_to(real)
    elif mutation == 'oversized': key.write_bytes(b'a' * 33)
    else:
        import os
        os.link(key, tmp_path / 'alias')
    with pytest.raises((OSError, ValueError)): review_key(key)
