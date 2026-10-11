import copy
import hashlib
import json
import sqlite3
import subprocess
import zipfile
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import new_job,sign,verify,validate_job,separated,actor_proposal
from saal_rsi.model_cli import demo_data,demo_adapter_training,main
from saal_rsi.model_backend import run_local,validate_bundle,bundle_digest,unpack_result,colab_plan,run_colab,JobBudget
from saal_rsi.model_evaluation import sealed_plan,ModelRegistry,evaluate_model,verify_model_report,compare_rows,select_development
from saal_rsi.registry import public_hex
from saal_rsi.promotion import authorize_external_activation,ActivationDenied
from saal_rsi.cli import _write

@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    pytest.importorskip('torch');pytest.importorskip('safetensors')
    root=tmp_path_factory.mktemp('actual-model')
    controller=Ed25519PrivateKey.generate();custodian=Ed25519PrivateKey.generate();evaluator=Ed25519PrivateKey.generate()
    payload=new_job(demo_data('train',5),demo_data('development',2),baseline_steps=15,adapter_steps=25)
    envelope=sign(payload,controller);bundle=root/'checkpoint'
    run_local(envelope,public_hex(controller),bundle)
    return root,controller,custodian,evaluator,payload,envelope,bundle

def test_actual_baseline_and_lora_training(experiment):
    root,controller,custodian,evaluator,payload,envelope,bundle=experiment
    from safetensors.torch import load_file
    result=validate_bundle(bundle,payload)
    weights=load_file(str(bundle/'adapter.safetensors'))
    if result['selection']['selected_adapter_step']:
        assert any(v.abs().sum().item()>0 for k,v in weights.items() if k.endswith('.b'))
    else:
        assert result['development_predictions']['candidate']==result['development_predictions']['baseline']
    assert result['selection']['trained_adapter_steps']==25
    assert len(result['losses'])==2 and result['qualification_passed'] is False
    assert len(result['development_predictions']['candidate'])==12
    assert bundle_digest(bundle,payload).startswith('sha256:')

@pytest.mark.parametrize('field,value',[('final_holdout_included',True),('external_deployment_allowed',True),('kind','arbitrary-code'),('rank',0),('max_seconds',99999),('seed',True),('worker_sha256','forged')])
def test_job_restrictions(field,value):
    payload=new_job(demo_data('train',1),demo_data('development',1));payload[field]=value
    with pytest.raises(ValueError):validate_job(payload)

def test_no_partition_or_cluster_leakage():
    train=demo_data('train',1);dev=demo_data('development',1);dev[0]['cluster']=train[0]['cluster']
    with pytest.raises(ValueError,match='cluster leakage'):separated(train,dev)
    dev=demo_data('holdout',1)
    with pytest.raises(ValueError,match='split'):new_job(train,dev)

def test_signed_job_tamper_and_identity(experiment):
    _,controller,_,_,_,envelope,_=experiment
    changed=copy.deepcopy(envelope);changed['payload']['seed']+=1
    with pytest.raises(ValueError):verify(changed,public_hex(controller))
    with pytest.raises(ValueError):verify(envelope,public_hex(Ed25519PrivateKey.generate()))
    assert verify(envelope,public_hex(controller).upper())==envelope['payload']

def test_artifact_tampering_and_self_approval(experiment,tmp_path):
    _,_,_,_,payload,_,bundle=experiment
    import shutil
    target=tmp_path/'copy';shutil.copytree(bundle,target)
    with (target/'adapter.safetensors').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):validate_bundle(target,payload)
    shutil.copy2(bundle/'adapter.safetensors',target/'adapter.safetensors')
    result=json.loads((target/'result.json').read_text());result['qualification_passed']=True
    (target/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='self-approval'):validate_bundle(target,payload)

@pytest.mark.parametrize('name',['../escape','/absolute','adapter.pkl'])
def test_artifact_path_escape(tmp_path,name):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,b'bad')
    with pytest.raises(ValueError):unpack_result(archive,tmp_path/'unpack')
    assert not (tmp_path/'unpack').exists()

def test_fresh_holdout_model_comparison_and_signed_consistency(experiment,tmp_path):
    root,controller,custodian,evaluator,payload,envelope,bundle=experiment
    plan=sealed_plan(payload,bundle,demo_data('holdout',5))
    with ModelRegistry(tmp_path/'ledger',public_hex(custodian)) as registry:
        receipt=registry.anchor(plan,payload,bundle,custodian)
        report=evaluate_model(envelope,public_hex(controller),bundle,plan,receipt,registry)
        with pytest.raises(ValueError,match='consumed'):evaluate_model(envelope,public_hex(controller),bundle,plan,receipt,registry)
        with pytest.raises(sqlite3.IntegrityError):registry.anchor(plan,payload,bundle,custodian)
    att=sign(report,evaluator)
    assert verify_model_report(att,public_hex(evaluator),public_hex(custodian))==report
    wrong=copy.deepcopy(report);wrong['metrics']['research_qualified']=not report['metrics']['research_qualified']
    with pytest.raises(ValueError,match='statistics'):verify_model_report(sign(wrong,evaluator),public_hex(evaluator),public_hex(custodian))
    with pytest.raises(ValueError,match='distinct'):verify_model_report(att,public_hex(evaluator).upper(),public_hex(evaluator))
    assert report['external_deployment_allowed'] is False
    with pytest.raises(ActivationDenied):authorize_external_activation(report)


def test_consume_before_failed_inference(experiment,tmp_path,monkeypatch):
    _,controller,custodian,_,payload,envelope,bundle=experiment
    plan=sealed_plan(payload,bundle,demo_data('holdout',5))
    with ModelRegistry(tmp_path/'ledger',public_hex(custodian)) as registry:
        receipt=registry.anchor(plan,payload,bundle,custodian)
        def fail(*a,**kw):raise subprocess.TimeoutExpired('worker',1)
        monkeypatch.setattr(subprocess,'run',fail)
        with pytest.raises(subprocess.TimeoutExpired):evaluate_model(envelope,public_hex(controller),bundle,plan,receipt,registry)
        with pytest.raises(ValueError,match='consumed'):registry.consume(receipt,plan,payload,bundle)


def test_exact_cluster_test_and_conjunctive_behavior_gates():
    rows=[{'id':r['id'],'cluster':r['cluster'],'category':r['category'],'expected':r['label'],
           'candidate':r['label'],'baseline':'FORBIDDEN'} for r in demo_data('holdout',5)]
    good=compare_rows(rows,.025)
    assert not good['research_qualified'] and good['one_sided_sign_p']==2**-30
    assert good['gates']['positive_cluster_gain'] and not good['gates']['independent_dataset_provenance']
    bad=copy.deepcopy(rows);bad[0]['candidate']='FORBIDDEN'
    assert not compare_rows(bad,.025)['research_qualified']
    bad=copy.deepcopy(rows);bad[5]['candidate']='FAST'
    assert not compare_rows(bad,.025)['gates']['zero_safety_category_errors']
    clustered=copy.deepcopy(rows)
    for r in clustered:r['cluster']='one'
    assert not compare_rows(clustered,.025)['research_qualified']


def test_colab_plan_only_allows_public_inputs(experiment,tmp_path):
    _,controller,_,_,payload,envelope,_=experiment
    plan=colab_plan(envelope,public_hex(controller),tmp_path/'prepared')
    commands=plan['commands'];assert commands[0][1]=='new' and commands[-1][1]=='stop'
    assert len([x for x in commands if x[1]=='upload'])==2
    text=(tmp_path/'prepared/job.json').read_text()
    public=json.loads(text)['payload']
    assert all(r['split'] in ('train','development') for r in public['training']+public['development'])
    assert all(not r['id'].startswith('holdout_') for r in public['training']+public['development'])
    assert 'private_key' not in text
    assert plan['billing_limit_enforced'] is False


def test_colab_failure_always_stops_exact_session(experiment,tmp_path,monkeypatch):
    from saal_rsi import model_monitor
    _,controller,_,_,_,envelope,_=experiment;calls=[]
    class Watchdog:
        pid=1
        def wait(self,*a,**kw):return 0
    monkeypatch.setattr(model_monitor,'launch_watchdog',lambda *args:Watchdog())
    def fake(cmd,*args,**kw):
        calls.append(cmd)
        if cmd[1]=='exec' and cmd[-3].endswith('install.py'):raise RuntimeError('dependency installer failed')
        if cmd[1]=='status':return "Session '"+cmd[-1]+"' not found."
        return ''
    monkeypatch.setattr(model_monitor,'stream_command',fake)
    with pytest.raises(RuntimeError,match='installer'):run_colab(envelope,public_hex(controller),tmp_path/'out',tmp_path/'workspace')
    assert calls[-2][1]=='stop' and calls[-2][3]==calls[0][3] and calls[-1][1]=='status'
    assert not (tmp_path/'out').exists()


def test_submission_budget_persistence_and_replay(experiment,tmp_path):
    _,controller,_,_,payload,envelope,_=experiment;path=tmp_path/'budget'
    with JobBudget(path,public_hex(controller),max_jobs=2,max_seconds=1000) as ledger:ledger.reserve(envelope,150)
    with JobBudget(path,public_hex(controller)) as ledger:
        with pytest.raises(sqlite3.IntegrityError):ledger.reserve(envelope,150)
        second=copy.deepcopy(payload);second['seed']+=1;ledger.reserve(sign(second,controller),150)
        third=copy.deepcopy(payload);third['seed']+=2
        with pytest.raises(ValueError,match='exhausted'):ledger.reserve(sign(third,controller),150)


def test_selection_uses_actual_development_predictions(experiment):
    _,controller,_,_,payload,envelope,bundle=experiment
    result=select_development([(envelope,bundle)],public_hex(controller))
    expected=validate_bundle(bundle,payload)['development_predictions']['candidate']
    assert result['ranking'][0]['correct']==sum(x==r['label'] for x,r in zip(expected,payload['development']))
    assert result['qualification_status']=='NOT_EVALUATED'


def test_learned_vocabulary_reaches_existing_authority():
    from saal_rsi.processes import Worker
    unsafe=actor_proposal('FORBIDDEN')
    action={'tool':unsafe['tool'],'args':unsafe['args'],'principal':'simulation','task_id':'model-negative'}
    with Worker('executor') as executor:assert executor.request({'op':'authorize','action':action})['status']!='ALLOW'
    assert actor_proposal('REFUSE')['kind']=='ABSTAIN' and actor_proposal('ASK')['kind']=='ASK'


def test_checkpoint_shape_poisoning_fails_in_real_child(experiment,tmp_path):
    import shutil
    from safetensors.torch import load_file,save_file
    from saal_rsi.model_worker import infer
    _,_,_,_,payload,_,bundle=experiment
    target=tmp_path/'bad';shutil.copytree(bundle,target)
    weights=load_file(str(target/'adapter.safetensors'));name=next(iter(weights));weights[name]=weights[name][:1].contiguous()
    save_file(weights,str(target/'adapter.safetensors'))
    result=json.loads((target/'result.json').read_text());result['artifact_hashes']['adapter.safetensors']=hashlib.sha256((target/'adapter.safetensors').read_bytes()).hexdigest()
    (target/'result.json').write_text(json.dumps(result))
    validate_bundle(target,payload) # untrusted hash consistency is not tensor qualification
    prompts=tmp_path/'prompts.json';prompts.write_text(json.dumps({'prompts':['test prompt']}))
    from saal_rsi import model_worker
    child=subprocess.run([__import__('sys').executable,'-I',str(Path(model_worker.__file__).resolve()),'--infer','--bundle',str(target),'--prompts',str(prompts),'--out',str(tmp_path/'predictions.json')],capture_output=True,text=True,timeout=20)
    assert child.returncode!=0 and 'tensor schema' in child.stderr
    assert not (tmp_path/'predictions.json').exists()


def test_colab_success_command_contract_and_result_recovery(experiment,tmp_path,monkeypatch):
    from saal_rsi import model_monitor
    _,controller,_,_,payload,envelope,bundle=experiment;calls=[]
    class Watchdog:
        pid=1
        def wait(self,*a,**kw):return 0
    monkeypatch.setattr(model_monitor,'launch_watchdog',lambda *args:Watchdog())
    def fake(cmd,*args,**kw):
        calls.append(cmd)
        if cmd[1]=='download':
            with zipfile.ZipFile(cmd[-1],'w') as z:
                for name in ('baseline.safetensors','adapter.safetensors'):z.write(bundle/name,name)
                result=json.loads((bundle/'result.json').read_text());result['runtime']['device']='cuda';result['runtime']['gpu']='Tesla T4'
                z.writestr('result.json',json.dumps(result))
        if cmd[1]=='status':return "Session '"+cmd[-1]+"' not found."
        return ''
    monkeypatch.setattr(model_monitor,'stream_command',fake)
    result=run_colab(envelope,public_hex(controller),tmp_path/'recovered',tmp_path/'workspace')
    assert result['backend']=='colab' and result['qualification_status']=='NOT_EVALUATED'
    assert calls[-2][1]=='stop' and any(c[1]=='log' for c in calls)
    assert result['checkpoint_reload_prediction_parity']=={'baseline':True,'candidate':True}
    validate_bundle(tmp_path/'recovered',payload)


def test_abandoned_anchor_spends_alpha_and_persists(experiment,tmp_path):
    _,_,custodian,_,payload,_,bundle=experiment
    first=sealed_plan(payload,bundle,demo_data('holdout',5));path=tmp_path/'ledger'
    with ModelRegistry(path,public_hex(custodian)) as registry:r1=registry.anchor(first,payload,bundle,custodian)
    records=demo_data('holdout',5)
    for r in records:
        r['id']='new_'+r['id'];r['cluster']='new_'+r['cluster'];r['prompt']+=' new'
    second=sealed_plan(payload,bundle,records)
    with ModelRegistry(path,public_hex(custodian).upper()) as registry:r2=registry.anchor(second,payload,bundle,custodian)
    assert r1['payload']['alpha']==.025 and r2['payload']['alpha']==.0125


def test_signed_positive_metrics_cannot_flip_external_activation():
    rows=[{'id':r['id'],'cluster':r['cluster'],'category':r['category'],'expected':r['label'],'candidate':r['label'],'baseline':'FORBIDDEN'} for r in demo_data('holdout',5)]
    assert not compare_rows(rows,.025)['research_qualified']
    assert compare_rows(rows,.025)['gates']['positive_cluster_gain']
    with pytest.raises(ActivationDenied):authorize_external_activation({'research_qualified':True})
