"""Exercise delivery signing, actual ZIP bytes, restoration and tamper rejection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def capsule(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir()
    project=root/'research/behavior-closure-v4';project.mkdir(parents=True)
    old=root/'research/reliable-learning-v3';old.mkdir()
    diverse=root/'research/diverse-learning-v1/experiment';diverse.mkdir(parents=True)
    (diverse/'signed_protocol.json').write_text('{}')
    (root/'src').mkdir();(root/'src/frozen.py').write_text('historical = True\n')
    frozen={'src/frozen.py':hashlib.sha256((root/'src/frozen.py').read_bytes()).hexdigest()}
    (old/'PREREGISTERED_FILES.json').write_text(json.dumps(frozen))
    (project/'colab-tests-02').mkdir();(project/'colab-tests-02/input-inventory.json').write_text('{}')
    (project/'RESULTS.json').write_text('{"runs":[]}')
    (project/'materialize_tensors.py').write_bytes((Path(__file__).resolve().parents[1]/'materialize_tensors.py').read_bytes())
    tensors=project/'artifacts';tensors.mkdir()
    # Large sparse fixture exercises real deduplication without allocating model memory.
    for name in ('initial','best'):
        with (tensors/(name+'.safetensors')).open('wb') as f:f.truncate(50_000_001)
    baseline=tmp_path/'baseline.zip'
    with zipfile.ZipFile(baseline,'w') as z:z.write(root/'src/frozen.py','src/frozen.py')
    key=Ed25519PrivateKey.generate();key_path=tmp_path/'controller.key'
    key_path.write_bytes(key.private_bytes_raw());key_path.chmod(0o600)
    package=load('package_capsule');monkeypatch.setattr(package,'ROOT',root);monkeypatch.setattr(package,'PROJECT',project)
    monkeypatch.setattr(package.subprocess,'check_output',lambda *a,**kw:'0'*40+'\n')
    archive=tmp_path/'delivery.zip';package.package(archive,str(key_path),baseline)
    extracted=tmp_path/'extracted';extracted.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    return root,project,extracted,key.public_key().public_bytes_raw().hex(),package,baseline,key_path


def test_signed_physical_capsule_restores_complete_checkpoint_paths(capsule):
    _,_,root,pin,_,_,_=capsule
    result=load('verify_capsule').verify_files(root,pin)
    assert result['checkpoint_paths_restored']==2
    paths=[root/'research/behavior-closure-v4/artifacts'/n for n in ('initial.safetensors','best.safetensors')]
    assert paths[0].stat().st_ino==paths[1].stat().st_ino
    assert paths[0].stat().st_size==50_000_001


@pytest.mark.parametrize('fault',['manifest','signature_pin','payload','extra','restoration_source'])
def test_delivery_tampering_never_verifies(capsule,fault):
    _,_,root,pin,_,_,_=capsule
    if fault=='manifest':
        with (root/'MANIFEST.sha256').open('a') as f:f.write('0'*64+'  unexpected.py\n')
    elif fault=='signature_pin':pin='0'*64
    elif fault=='payload':next((root/'tensor_payloads').iterdir()).write_bytes(b'corrupted')
    elif fault=='extra':(root/'unexpected.py').write_text('extra')
    else:(root/'materialize_tensors.py').write_text('raise RuntimeError("untrusted code")')
    with pytest.raises((ValueError,PermissionError)):
        load('verify_capsule').verify_files(root,pin)


@pytest.mark.parametrize('fault',['lfs_pointer','historical_source'])
def test_packaging_rejects_incomplete_or_rewritten_history(capsule,tmp_path,fault):
    root,project,_,_,package,baseline,key=capsule
    if fault=='lfs_pointer':(project/'missing.safetensors').write_text('version https://git-lfs.github.com/spec/v1\n')
    else:(root/'src/frozen.py').write_text('historical = False\n')
    with pytest.raises(ValueError):package.package(tmp_path/'bad.zip',str(key),baseline)
