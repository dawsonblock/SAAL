import copy
import json
import shutil
from pathlib import Path
import pytest
from saal_rsi.spec import parse_unique_json,canonical
from saal_rsi.pretrained_artifacts import validate_bundle,seal

@pytest.fixture
def bundle(tmp_path):
    root=Path(__file__).resolve().parents[2];example=root/'examples/pretrained-head-compat-v2'
    if not example.exists():pytest.skip('release example not materialized')
    shutil.copytree(example,tmp_path/'bundle');out=tmp_path/'bundle';job=parse_unique_json((out/'signed_training_job.json').read_bytes())['payload'];pin=(root/'examples/pretrained-controller.pub').read_text().strip();return out,job,pin

def test_real_gpu_bundle_and_negative_findings(bundle):
    out,j,pin=bundle;r=validate_bundle(out,j,pin)
    assert r['runtime']['device']=='cuda:0' and r['real_parameter_updates'] is True
    assert r['snapshots']['best']['metrics']['development']['class_accuracy']['FAST']==0
    assert r['snapshots']['best']['metrics']['development']['complete_pair_accuracy']==0
    assert r['qualification_status']=='NOT_EVALUATED'

def test_tampered_tensor_inventory(bundle):
    from safetensors.torch import load_file,save_file
    import torch
    out,j,pin=bundle;s=load_file(str(out/'best.safetensors'));s['classifier.weight']=torch.zeros((5,768));save_file(s,str(out/'best.safetensors'));seal(out)
    with pytest.raises(ValueError):validate_bundle(out,j,pin)

def test_nonfinite_adapter(bundle):
    from safetensors.torch import load_file,save_file
    out,j,pin=bundle;s=load_file(str(out/'terminal.safetensors'));s['classifier.weight'][0,0]=float('nan');save_file(s,str(out/'terminal.safetensors'));seal(out)
    with pytest.raises(ValueError):validate_bundle(out,j,pin)

@pytest.mark.parametrize('mutation',['metrics','identity','runtime','promotion','selection','work'])
def test_forged_evidence_and_authority(bundle,mutation):
    out,j,pin=bundle;r=parse_unique_json((out/'result.json').read_bytes())
    if mutation=='metrics':r['snapshots']['best']['metrics']['development']['accuracy']=1.0
    elif mutation=='identity':r['restoration_manifest']['head_digest']='sha256:'+'0'*64
    elif mutation=='runtime':r['runtime']['versions']['transformers']='999.0'
    elif mutation=='promotion':r['external_deployment_allowed']=True
    elif mutation=='selection':r['snapshots']['best']['step']=0
    else:r['resource_accounting']['steps']=10000
    (out/'result.json').write_bytes(canonical(r));seal(out)
    with pytest.raises(ValueError):validate_bundle(out,j,pin)

def test_cross_job_substitution(bundle):
    out,j,pin=bundle;other=copy.deepcopy(j);other['seed']=29
    with pytest.raises(ValueError):validate_bundle(out,other,pin)

def test_checkpoint_symlink(bundle,tmp_path):
    out,j,pin=bundle;(out/'best.safetensors').rename(tmp_path/'replacement');(out/'best.safetensors').symlink_to(tmp_path/'replacement')
    with pytest.raises(ValueError):validate_bundle(out,j,pin)
