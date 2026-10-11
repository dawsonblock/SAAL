from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import sign
from saal_rsi.neural_contract import new_transformer_job
from saal_rsi.research_datasets import semantic_benchmark
from saal_rsi.neural_backend import neural_colab_plan
from saal_rsi.neural_campaign import campaign_protocol
from saal_rsi.experiment_registry import ExperimentRegistry

def test_hashed_gpu_installer_and_named_cleanup(tmp_path):
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex();job=new_transformer_job(semantic_benchmark(2,1),baseline_steps=1,adaptation_steps=1)
    plan=neural_colab_plan(sign(job,key),pin,tmp_path/'remote')
    assert plan['dependency_hash_lock_enforced']
    installer=(tmp_path/'remote/install.py').read_text();assert '--require-hashes' in installer and job['dependency_lock_sha256'] in installer
    assert plan['commands'][-1]==['colab','stop','-s',plan['session']]
    assert plan['includes_final_holdout'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'remote/worker.zip') as z:
        from saal_rsi.neural_contract import SOURCES
        assert set(z.namelist())=={'worker.py','saal_rsi/__init__.py',*('saal_rsi/'+n for n in SOURCES)}
        assert 'saal_rsi/model_evaluation.py' not in z.namelist()
        assert 'saal_rsi/promotion.py' not in z.namelist()

def test_registry_preserves_negative_and_denies_restart(tmp_path):
    protocol=campaign_protocol(semantic_benchmark(2,1),seeds=[17],baseline_steps=1,adaptation_steps=1)
    with ExperimentRegistry(tmp_path/'registry.db') as registry:
        digest=registry.register(protocol);registry.start(digest);registry.finish(digest,{'error':'timeout'},completed=False)
        assert registry.records()[0]['state']=='INTERRUPTED'
        with pytest.raises(ValueError):registry.start(digest)
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):registry.register(protocol)


def test_campaign_admission_rejects_unsafe_modes_and_budgets(tmp_path):
    protocol=campaign_protocol(semantic_benchmark(2,1),seeds=[17],baseline_steps=1,adaptation_steps=1)
    with ExperimentRegistry(tmp_path/'registry.db') as registry:
        protocol['max_run_seconds']=True
        with pytest.raises(ValueError):registry.register(protocol)
        protocol['max_run_seconds']=100;protocol['modes'].append('promote')
        with pytest.raises(ValueError):registry.register(protocol)


def test_real_local_registered_campaign_keeps_identical_backbones(tmp_path):
    import hashlib,json,os
    from saal_rsi.neural_campaign import execute_campaign
    key=Ed25519PrivateKey.generate();private=tmp_path/'controller.key';private.write_bytes(key.private_bytes_raw());os.chmod(private,0o600)
    dataset=semantic_benchmark(2,1);protocol=campaign_protocol(dataset,seeds=[17],baseline_steps=1,adaptation_steps=1,max_run_seconds=120)
    pin=key.public_key().public_bytes_raw().hex()
    report=execute_campaign(sign(protocol,key),pin,private,dataset,tmp_path/'campaign')
    assert len(report['runs'])==4
    assert len({r['baseline_sha256'] for r in report['runs']})==1
    assert report['qualification_status']=='NOT_EVALUATED'
    assert report['independent_generalization']=='NOT_ACHIEVED'


def test_baseline_upload_and_training_output_are_retained(tmp_path):
    import hashlib
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex();baseline=tmp_path/'baseline.safetensors';baseline.write_bytes(b'bounded-fixture')
    job=new_transformer_job(semantic_benchmark(2,1),baseline_steps=0,adaptation_steps=1,baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    plan=neural_colab_plan(sign(job,key),pin,tmp_path/'remote',baseline)
    assert 'UPLOAD_BASELINE' in plan['phases']
    upload=plan['commands'][plan['phases'].index('UPLOAD_BASELINE')]
    assert upload[1]=='upload' and upload[-1]=='/content/saal_baseline.safetensors'
    assert 'subprocess.PIPE' in (tmp_path/'remote/run.py').read_text()


def test_failed_job_is_recorded_without_parser_masking(tmp_path,monkeypatch):
    import json,os
    import saal_rsi.neural_campaign as campaign
    key=Ed25519PrivateKey.generate();private=tmp_path/'controller.key';private.write_bytes(key.private_bytes_raw());os.chmod(private,0o600)
    pin=key.public_key().public_bytes_raw().hex();dataset=semantic_benchmark(2,1)
    protocol=campaign_protocol(dataset,seeds=[17],baseline_steps=1,adaptation_steps=1,max_run_seconds=120)
    def failed(*a,**k):raise RuntimeError('training failure sentinel')
    monkeypatch.setattr(campaign,'run_neural_local',failed)
    root=tmp_path/'campaign'
    with pytest.raises(RuntimeError,match='sentinel'):campaign.execute_campaign(sign(protocol,key),pin,private,dataset,root)
    assert json.loads((root/'run-outcomes.json').read_text())['runs'][0]['status']=='INTERRUPTED'
    with ExperimentRegistry(root/'experiments.db') as registry:
        assert registry.records()[0]['state']=='INTERRUPTED'
        assert registry.records()[0]['outcome']['error']['detail']=='training failure sentinel'


def test_live_credential_is_removed_from_stream_logs_and_error(tmp_path):
    import sys
    from saal_rsi.model_monitor import stream_command
    credential='monitor-secret-sentinel'
    code="import sys;sys.stdout.write('https://host/path?colab-runtime-proxy-token="+credential+"&content=1\\n');sys.exit(1)"
    log=tmp_path/'command.log'
    with pytest.raises(RuntimeError) as exc:stream_command([sys.executable,'-c',code],log,10)
    assert credential not in str(exc.value)
    assert credential not in log.read_text()
    assert '[REDACTED]' in log.read_text()


def test_transfer_retries_do_not_authorize_training_retries(tmp_path):
    protocol=campaign_protocol(semantic_benchmark(2,1),seeds=[17],baseline_steps=1,adaptation_steps=1)
    assert protocol['transfer_retries']==2
    protocol['transfer_retries']=100
    with ExperimentRegistry(tmp_path/'registry.db') as registry:
        with pytest.raises(ValueError,match='transfer'):registry.register(protocol)


def test_duplicate_seed_metadata_cannot_fake_replication():
    from saal_rsi.neural_campaign import campaign_summary
    runs=[{'seed':17,'mode':'frozen','status':'INTERRUPTED'}]*2
    with pytest.raises(ValueError,match='duplicate'):campaign_summary(runs,'unused')
