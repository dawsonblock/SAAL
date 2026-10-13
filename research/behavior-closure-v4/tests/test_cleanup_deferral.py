import pytest
from saal_rsi import epoch_backend_v4 as backend
from pathlib import Path
import json,hashlib,zipfile,shutil
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.model_contract import verify
from saal_rsi.spec import sha


def test_optional_export_cleanup_timeout_is_reported_without_training_retry(monkeypatch,tmp_path):
    calls=[]
    def timeout(command,log,seconds):
        calls.append(command);raise TimeoutError('transport deadline')
    monkeypatch.setattr(backend,'stream_command',timeout)
    result=backend.cleanup_export(['colab','exec','-s','exact-session'],tmp_path/'cleanup.log',40)
    assert result['state']=='DEFERRED' and result['deferred_to_session_stop'] is True
    assert len(calls)==1


def test_successful_cleanup_has_no_deferred_claim(monkeypatch,tmp_path):
    monkeypatch.setattr(backend,'stream_command',lambda *args:'')
    assert backend.cleanup_export(['colab','exec'],tmp_path/'log',40)=={'state':'CONFIRMED','deferred_to_session_stop':False}


def test_invalid_or_reused_local_log_is_not_masked_as_optional_network_failure(monkeypatch,tmp_path):
    def collision(*args):raise FileExistsError('preserve previous evidence')
    monkeypatch.setattr(backend,'stream_command',collision)
    with pytest.raises(FileExistsError):backend.cleanup_export(['colab','exec'],tmp_path/'log',40)


@pytest.mark.parametrize('corrupt',[False,True])
def test_real_native_validation_precedes_cleanup_deferral_and_corruption_still_interrupts(monkeypatch,tmp_path,corrupt):
    repo=Path(__file__).resolve().parents[3]
    project=repo/'research/reliable-learning-v3'
    source=project/'experiment/artifacts/run-01-c6ae464acc2c'
    pin=(project/'experiment/controller.pub').read_text().strip()
    envelope=json.loads((source/'signed_training_job.json').read_text());job=verify(envelope,pin)
    workspace=tmp_path/'controller';workspace.mkdir();artifacts=tmp_path/'recovered';artifacts.mkdir()
    lock=repo/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt'
    shutil.copyfile(lock,workspace/'gpu-lock.txt');job_path=workspace/'job.json';job_path.write_text(json.dumps(envelope))
    ledger_path=workspace/'budget.sqlite3';ledger=EpochExperimentRegistry(ledger_path,pin,max_jobs=1,max_seconds=3800)
    ledger.reserve(envelope);ledger.close()
    archive=tmp_path/'native.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(source.iterdir()):z.write(p,p.name)
    raw=archive.read_bytes();chunk=tmp_path/'chunk';chunk.write_bytes(raw)
    h=hashlib.sha256(raw).hexdigest()
    manifest={'schema':'saal-artifact-transfer-v1','job_digest':sha(job),'size':len(raw),'sha256':h,
              'chunks':[{'name':'part-000.bin','size':len(raw),'sha256':h}]}
    monkeypatch.setattr(backend,'run_remote_stage',lambda *args:manifest)
    def download(command,*args):
        Path(command[-1]).write_bytes(raw if not corrupt else raw[:-1]+bytes([raw[-1]^1]))
    monkeypatch.setattr(backend,'download_read',download)
    cleanup_calls=[]
    def cleanup_timeout(command,*args):
        if command[1]=='exec':
            cleanup_calls.append(command);raise TimeoutError('optional export cleanup deadline')
        return ''
    monkeypatch.setattr(backend,'stream_command',cleanup_timeout)
    config=workspace/'config.json';config.write_text(json.dumps({'workspace':str(workspace),'session':'test-no-remote','pin':pin,
             'jobs':[str(job_path)],'artifacts':str(artifacts),'ledger':str(ledger_path)}))
    if corrupt:
        with pytest.raises(RuntimeError):backend.coordinator(config)
    else:backend.coordinator(config)
    rows=json.loads((workspace/'outcomes.json').read_text())['runs']
    assert rows[0]['state']==('INTERRUPTED' if corrupt else 'COMPLETED')
    assert len(cleanup_calls)==(0 if corrupt else 1)
    if not corrupt:
        assert rows[0]['artifact_cleanup']['state']=='DEFERRED'
        assert rows[0]['result']['job_digest']==sha(job)
    ledger=EpochExperimentRegistry(ledger_path,pin)
    assert ledger.records()['attempts'][0]['state']==rows[0]['state'];ledger.close()
