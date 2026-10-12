"""Real native checkpoint evidence plus substitution and forgery negatives."""
import json,shutil
from pathlib import Path
import pytest
from saal_rsi.spec import canonical
from saal_rsi.epoch_artifacts import validate_bundle,seal

@pytest.fixture
def bundle(tmp_path):
    root=Path(__file__).resolve().parents[2];p=root/'research/reliable-learning-v3'
    shutil.copytree(p/'local-smoke-03',tmp_path/'bundle');out=tmp_path/'bundle'
    job=json.loads((out/'signed_training_job.json').read_text())['payload'];pin=(p/'controller.pub').read_text().strip()
    return out,job,pin

def test_real_native_v6_smoke_and_no_promotion(bundle):
    p,j,pin=bundle;r=validate_bundle(p,j,pin)
    assert r['real_parameter_updates'] and r['reload_parity']
    assert r['qualification_status']=='NOT_EVALUATED' and not r['external_deployment_allowed']
    assert len(r['full_training_evaluation']['terminal']['predictions'])==400

@pytest.mark.parametrize('mutation',['train_metrics','full_train','sampler','diagnostic','rate','updates','tokens','warmup','authority','tensor'])
def test_artifact_forgery_rejected(bundle,mutation):
    p,j,pin=bundle;r=json.loads((p/'result.json').read_text())
    if mutation=='train_metrics':r['snapshots']['terminal']['metrics']['train']['accuracy']=1.0
    elif mutation=='full_train':r['full_training_evaluation']['terminal']['predictions'].pop()
    elif mutation=='sampler':
        s=json.loads((p/'sampler_manifest.json').read_text());s['record_ids'][0]=next(x['record_id'] for x in j['dataset']['records'] if x['split']=='regression');(p/'sampler_manifest.json').write_bytes(canonical(s))
    elif mutation=='diagnostic':
        s=json.loads((p/'diagnostic_manifest.json').read_text());s['record_ids'][0]=s['record_ids'][1];(p/'diagnostic_manifest.json').write_bytes(canonical(s))
    elif mutation in ('rate','updates'):
        rows=[json.loads(s) for s in (p/'training_metrics.jsonl').read_text().splitlines()]
        for s in rows:
            if s['monitor']=='training_progress':
                if mutation=='rate':s['optimizer_groups']['head']=.1
                else:s['completed_steps']=99
                break
        (p/'training_metrics.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in rows))
    elif mutation=='tokens':r['resource_accounting']['tokens']+=1
    elif mutation=='warmup':r['warmup_evidence']['encoder_unchanged']=True
    elif mutation=='authority':r['external_deployment_allowed']=True
    else:
        from safetensors.torch import load_file,save_file
        s=load_file(str(p/'terminal.safetensors'));s['classifier.weight'][0,0]=float('nan');save_file(s,str(p/'terminal.safetensors'))
    (p/'result.json').write_bytes(canonical(r));seal(p)
    with pytest.raises(ValueError):validate_bundle(p,j,pin)
