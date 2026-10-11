import copy
import json
import subprocess
import sys
import time
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import new_job,sign,validate_job
from saal_rsi.model_cli import demo_data,demo_adapter_training
from saal_rsi.model_backend import run_local,validate_bundle,colab_plan
from saal_rsi.model_evaluation import sealed_plan
from saal_rsi.model_worker import development_score
from saal_rsi.registry import public_hex


def test_adapter_partition_leakage_is_rejected():
    train=demo_data('train',1);dev=demo_data('development',1)
    adapter=copy.deepcopy(dev)
    for r in adapter:r['split']='train'
    with pytest.raises(ValueError,match='leakage'):new_job(train,dev,adapter_training=adapter)
    adapter=demo_data('holdout',1)
    with pytest.raises(ValueError,match='split'):new_job(train,dev,adapter_training=adapter)


def test_final_holdout_excludes_adapter_training_too(tmp_path):
    train=demo_data('train',1);dev=demo_data('development',1);holdout=demo_data('holdout',5)
    adapter=copy.deepcopy(holdout)
    for r in adapter:r['split']='train'
    payload=new_job(train,dev,adapter_training=adapter)
    with pytest.raises(ValueError,match='leakage'):sealed_plan(payload,tmp_path/'absent',holdout)

@pytest.mark.parametrize('field,value',[('adapter_learning_rate',0.0),('adapter_learning_rate',float('nan')),('adapter_objective','unweighted'),('checkpoint_selection','last-step')])
def test_learning_parameters_are_closed(field,value):
    payload=new_job(demo_data('train',1),demo_data('development',1));payload[field]=value
    with pytest.raises(ValueError):validate_job(payload)


def test_augmented_adapter_improves_development_without_changing_baseline(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('safetensors')
    train=demo_data('train',20);dev=demo_data('development',8);key=Ed25519PrivateKey.generate()
    original=new_job(train,dev,adapter_steps=1)
    augmented=new_job(train,dev,adapter_training=demo_adapter_training(train))
    results=[]
    for name,payload in [('incumbent',original),('augmented',augmented)]:
        out=tmp_path/name;run_local(sign(payload,key),public_hex(key),out);results.append(validate_bundle(out,payload))
    first,second=results
    assert first['artifact_hashes']['baseline.safetensors']==second['artifact_hashes']['baseline.safetensors']
    assert first['development_predictions']['baseline']==second['development_predictions']['baseline']
    assert second['selection']['development_score']>list(development_score(first['development_predictions']['baseline'],dev))
    assert second['selection']['selected_adapter_step']>0
    assert second['selection']['development_score']==max([list(development_score(second['development_predictions']['baseline'],dev))]+[h['development_score'] for h in second['training_history'] if h['phase']=='adapter'])
    assert second['selection']['labels_absent_from_training']==['FAST']
    bad=copy.deepcopy(second);bad['selection']['selected_adapter_step']=201
    (tmp_path/'augmented/result.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='checkpoint step'):validate_bundle(tmp_path/'augmented',augmented)


def test_monitor_streams_actual_child_events_and_kills_stalls(tmp_path):
    from saal_rsi.model_monitor import stream_command
    events=[]
    stream_command([sys.executable,'-c',"import json; print(json.dumps({'monitor':'test','step':1}))"],tmp_path/'good.log',3,events.append)
    assert events==[{'monitor':'test','step':1}]
    started=time.monotonic()
    with pytest.raises(TimeoutError):stream_command([sys.executable,'-c','import time; time.sleep(20)'],tmp_path/'stall.log',.1)
    assert time.monotonic()-started<3


def test_watchdog_refuses_changed_session_identity(tmp_path):
    from saal_rsi.model_monitor import watchdog
    state=tmp_path/'state.json';state.write_text(json.dumps({'session':'unrelated','terminal':False}))
    with pytest.raises(ValueError,match='session binding'):watchdog(state,time.time()-1,'saal-'+'a'*20)


def test_dependency_plan_has_heartbeats_and_explicit_timeout(tmp_path):
    key=Ed25519PrivateKey.generate();payload=new_job(demo_data('train',1),demo_data('development',1))
    plan=colab_plan(sign(payload,key),public_hex(key),tmp_path/'public')
    installer=(tmp_path/'public/install.py').read_text()
    assert 'dependency_heartbeat' in installer and 'timeout=300' in installer
    assert plan['commands'][1][1]=='exec' and plan['commands'][1][-1]=='330'
    assert '--require-gpu' in (tmp_path/'public/run.py').read_text()


def test_cleanup_requires_confirmation_not_only_stop_success(tmp_path,monkeypatch):
    from saal_rsi import model_monitor
    key=Ed25519PrivateKey.generate();payload=new_job(demo_data('train',1),demo_data('development',1))
    plan=colab_plan(sign(payload,key),public_hex(key),tmp_path/'public')
    class Watchdog:
        pid=1
        def wait(self,*args,**kwargs):return 0
    monkeypatch.setattr(model_monitor,'launch_watchdog',lambda *args:Watchdog())
    calls=[]
    def fake(command,*args,**kwargs):calls.append(command);return 'Still active'
    monkeypatch.setattr(model_monitor,'stream_command',fake)
    with pytest.raises(RuntimeError,match='cleanup failed'):model_monitor.execute_colab_plan(plan,tmp_path/'public')
    state=json.loads((tmp_path/'public/monitor-state.json').read_text())
    assert state['phase']=='FAILED' and state['cleanup_verified'] is False
    assert calls[-2][1]=='stop' and calls[-1][1]=='status'


def test_frozen_baseline_requires_zero_steps_and_valid_pin():
    train=demo_data('train',1);dev=demo_data('development',1)
    with pytest.raises(ValueError,match='requires training'):new_job(train,dev,baseline_steps=0)
    with pytest.raises(ValueError,match='frozen baseline'):new_job(train,dev,baseline_sha256='a'*64)
    with pytest.raises(ValueError,match='frozen baseline'):new_job(train,dev,baseline_steps=0,baseline_sha256='bad')


def test_frozen_baseline_input_refuses_wrong_hash_and_links(tmp_path):
    from saal_rsi.model_contract import baseline_input
    p=tmp_path/'base';p.write_bytes(b'checkpoint')
    with pytest.raises(ValueError,match='hash mismatch'):baseline_input(p,'a'*64)
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(OSError):baseline_input(link)


def test_frozen_baseline_is_preserved_and_substitution_rejected(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('safetensors')
    key=Ed25519PrivateKey.generate();pin=public_hex(key)
    train=demo_data('train',1);dev=demo_data('development',1)
    original=new_job(train,dev,baseline_steps=1,adapter_steps=1)
    first=tmp_path/'first';run_local(sign(original,key),pin,first)
    sha=validate_bundle(first,original)['artifact_hashes']['baseline.safetensors']
    frozen=new_job(train,dev,baseline_steps=0,adapter_steps=1,baseline_sha256=sha)
    second=tmp_path/'second';run_local(sign(frozen,key),pin,second,first/'baseline.safetensors')
    assert (second/'baseline.safetensors').read_bytes()==(first/'baseline.safetensors').read_bytes()
    assert validate_bundle(second,frozen)['artifact_hashes']['baseline.safetensors']==sha
    broken=tmp_path/'broken';broken.write_bytes(b'invalid')
    with pytest.raises(ValueError,match='hash mismatch'):run_local(sign(frozen,key),pin,tmp_path/'reject',broken)
