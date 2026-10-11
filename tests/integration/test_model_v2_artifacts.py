import copy,json,shutil
from pathlib import Path
import pytest,torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file,save_file
from saal_rsi.neural_contract import ModelSpec,new_transformer_job
from saal_rsi.research_datasets import semantic_benchmark
from saal_rsi.model_contract import sign
from saal_rsi.model_backend import run_local,validate_bundle
from saal_rsi.neural_artifacts import validate_neural_bundle
from saal_rsi.neural_worker import seal_inventory

@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    root=tmp_path_factory.mktemp('neural');key=Ed25519PrivateKey.generate()
    spec=ModelSpec(hidden_dimension=32,encoder_layers=1,feedforward_dimension=128,dropout=0.0).as_dict()
    job=new_transformer_job(semantic_benchmark(2,1),model_spec=spec,baseline_steps=2,adaptation_steps=2,max_seconds=120)
    envelope=sign(job,key);pin=key.public_key().public_bytes_raw().hex();bundle=root/'bundle'
    run_local(envelope,pin,bundle);return root,job,envelope,pin,bundle

def test_real_child_reload(artifact):
    _,job,envelope,pin,bundle=artifact
    result=validate_neural_bundle(bundle,job,replay=True)
    assert result['schema']=='saal-model-result-v3'
    assert result['qualification_status']=='NOT_EVALUATED'
    assert result['backbone_unchanged'] is True

def test_adapter_shape_and_weight_tamper_after_rehash(artifact,tmp_path):
    _,job,_,_,bundle=artifact;out=tmp_path/'copy';shutil.copytree(bundle,out)
    adapter=load_file(str(out/'adapter.safetensors'));name=next(iter(adapter));adapter[name]=adapter[name][:1]
    save_file(adapter,str(out/'adapter.safetensors'));seal_inventory(out,json.loads((out/'result.json').read_text()))
    with pytest.raises(ValueError,match='tensor|adapter'):validate_bundle(out,job)

def test_backbone_mutation_with_matching_candidate_and_rehash(artifact,tmp_path):
    _,job,_,_,bundle=artifact;out=tmp_path/'copy';shutil.copytree(bundle,out)
    candidate=load_file(str(out/'candidate.safetensors'));candidate['head.bias'][0]+=1
    save_file(candidate,str(out/'candidate.safetensors'));seal_inventory(out,json.loads((out/'result.json').read_text()))
    with pytest.raises(ValueError,match='backbone'):validate_bundle(out,job)

@pytest.mark.parametrize('mutation',['nan','prediction','self-approval','extra','partial','spec','signed-job'])
def test_invalid_evidence(artifact,tmp_path,mutation):
    _,job,_,_,bundle=artifact;out=tmp_path/'copy';shutil.copytree(bundle,out);result=json.loads((out/'result.json').read_text())
    if mutation=='nan':result['development_predictions']['candidate']['logits'][0][0]=float('nan');(out/'result.json').write_text(json.dumps(result))
    elif mutation=='prediction':
        result['development_predictions']['candidate']['labels'][0]='FORBIDDEN';seal_inventory(out,result)
    elif mutation=='self-approval':result['qualification_passed']=True;seal_inventory(out,result)
    elif mutation=='extra':(out/'surprise.py').write_text('')
    elif mutation=='partial':(out/'candidate.safetensors').unlink()
    elif mutation=='spec':(out/'model_spec.json').write_text('{}');seal_inventory(out,result)
    else:(out/'signed_training_job.json').write_text('{}');seal_inventory(out,result)
    with pytest.raises((ValueError,KeyError)):validate_bundle(out,job)

def test_same_baseline_bytes_across_rank_and_replay(artifact,tmp_path):
    _,job,_,_,bundle=artifact
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex()
    import hashlib
    spec=copy.deepcopy(job['model_spec']);spec['adapter_configuration']['rank']=8
    new=new_transformer_job(job['dataset'],model_spec=spec,baseline_steps=0,adaptation_steps=1,baseline_sha256=hashlib.sha256((bundle/'baseline.safetensors').read_bytes()).hexdigest(),max_seconds=120)
    out=tmp_path/'rank8';run_local(sign(new,key),pin,out,bundle/'baseline.safetensors')
    assert (out/'baseline.safetensors').read_bytes()==(bundle/'baseline.safetensors').read_bytes()
    validate_neural_bundle(out,new,replay=True)


def test_v4_family_custody_and_one_use_negative_qualification(artifact,tmp_path):
    import hashlib
    from saal_rsi.spec import sha
    from saal_rsi.dataset_registry import manifest
    from saal_rsi.family_evaluation import create_review,family_plan
    from saal_rsi.model_evaluation import ModelRegistry,evaluate_model,verify_model_report
    _,job,envelope,controller_pin,bundle=artifact
    records=copy.deepcopy([r for r in job['dataset']['records'] if r['split']=='regression'])
    for r in records:
        for field in ('record_id','task_family_id','cluster_id','template_id'):r[field]='sealed_'+r[field]
        if r['parent_record_id']:r['parent_record_id']='sealed_'+r['parent_record_id']
        r['split']='holdout';r['prompt']=r['prompt'].replace('request regression','request sequestered')
        r['source_digest']=sha({'public-fixture':r['source_digest']})
    holdout=manifest(records);custodian=Ed25519PrivateKey.generate();evaluator=Ed25519PrivateKey.generate();custodian_pin=custodian.public_key().public_bytes_raw().hex()
    review=create_review(job['dataset'],holdout,custodian)
    plan=family_plan(job,bundle,job['dataset'],holdout,review,custodian_pin,controller_pin)
    with ModelRegistry(tmp_path/'custody.db',custodian_pin) as registry:
        receipt=registry.anchor(plan,job,bundle,custodian)
        report=evaluate_model(envelope,controller_pin,bundle,plan,receipt,registry)
        assert report['metrics']['research_qualified'] is False
        assert report['external_deployment_allowed'] is False
        verified=verify_model_report(sign(report,evaluator),evaluator.public_key().public_bytes_raw().hex(),custodian_pin)
        assert verified['model_activation_allowed'] is False
        with pytest.raises(ValueError,match='consumed'):evaluate_model(envelope,controller_pin,bundle,plan,receipt,registry)


def test_controlled_cpu_training_reproduces_checkpoint_bytes(artifact,tmp_path):
    _,job,envelope,pin,bundle=artifact
    other=tmp_path/'repeat';run_local(envelope,pin,other)
    for name in ('baseline.safetensors','adapter.safetensors','candidate.safetensors'):
        assert (bundle/name).read_bytes()==(other/name).read_bytes()
