import hashlib,json,time
from pathlib import Path
import pytest
from saal_rsi import epoch_backend_v2 as backend
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.model_contract import verify

def manifest():
 h=hashlib.sha256(b'data').hexdigest();return {'schema':'saal-artifact-transfer-v1','job_digest':'sha256:'+'1'*64,'size':4,'sha256':h,'chunks':[{'name':'part-000.bin','size':4,'sha256':h}]}

def test_read_retry_recovers_without_retraining(monkeypatch,tmp_path):
 calls=[]
 def command(args,*unused):
  calls.append(args)
  if len(calls)<3:raise RuntimeError('file endpoint not yet visible')
  Path(args[-1]).write_bytes(b'data')
 monkeypatch.setattr(backend,'stream_command',command);monkeypatch.setattr(backend.time,'sleep',lambda n:None)
 out=tmp_path/'part.bin';backend.download_read(['colab','download','-s','owned','remote',str(out)],tmp_path/'read.log',time.monotonic()+30,manifest()['chunks'][0])
 assert out.read_bytes()==b'data' and len(calls)==3
 assert all(c[:2]==['colab','download'] for c in calls)

@pytest.mark.parametrize('mode',['notfound','corrupt'])
def test_read_failure_is_bounded_and_never_accepted(monkeypatch,tmp_path,mode):
 calls=[]
 def command(args,*unused):
  calls.append(args)
  if mode=='notfound':raise RuntimeError('missing')
  Path(args[-1]).write_bytes(b'wrong')
 monkeypatch.setattr(backend,'stream_command',command);monkeypatch.setattr(backend.time,'sleep',lambda n:None)
 out=tmp_path/'part.bin'
 with pytest.raises(RuntimeError):backend.download_read(['colab','download','-s','owned','remote',str(out)],tmp_path/'read.log',time.monotonic()+30,manifest()['chunks'][0])
 assert len(calls)==5 and not out.exists()

def test_retry_cannot_execute_a_training_command(tmp_path):
 with pytest.raises(ValueError):backend.download_read(['colab','exec'],tmp_path/'log',time.monotonic()+30,manifest()['chunks'][0])

@pytest.mark.parametrize('events',[[],[{'monitor':'artifact_transfer_ready','manifest':manifest()}]*2,[{'monitor':'worker_failure','stage':'export'}]])
def test_incomplete_or_duplicate_export_acknowledgement_fails(monkeypatch,tmp_path,events):
 def command(args,log,timeout,callback):
  for e in events:callback(e)
  return 'CLI exit zero'
 monkeypatch.setattr(backend,'stream_command',command)
 with pytest.raises(RuntimeError):backend.run_remote_stage(['colab','exec'],tmp_path/'log',30)

def test_acknowledgement_carries_validated_manifest(monkeypatch,tmp_path):
 def command(args,log,timeout,callback):callback({'monitor':'artifact_transfer_ready','manifest':manifest()});return 'ok'
 monkeypatch.setattr(backend,'stream_command',command)
 assert backend.run_remote_stage(['colab','exec'],tmp_path/'log',30)==manifest()

def test_generated_remote_program_is_valid_and_guards_export(monkeypatch,tmp_path):
 root=Path(__file__).resolve().parents[2];project=root/'research/reliable-learning-v3';envelope=json.loads((project/'experiment/uniform-v1_109-job.json').read_text());pin=(project/'controller.pub').read_text().strip();job=verify(envelope,pin)
 (tmp_path/'gpu-lock.txt').write_bytes((root/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt').read_bytes());job_path=tmp_path/'job.json';job_path.write_text(json.dumps(envelope));ledger=tmp_path/'ledger.sqlite'
 reg=EpochExperimentRegistry(ledger,pin,max_jobs=1,max_seconds=3800);reg.reserve(envelope);reg.close()
 config={'workspace':str(tmp_path),'session':'saal-'+'a'*20,'pin':pin,'ledger':str(ledger),'jobs':[str(job_path)],'artifacts':str(tmp_path/'artifacts')};cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config));seen=[]
 monkeypatch.setattr(backend,'stream_command',lambda *a,**k:'upload')
 def probe(command,log,timeout):
  script=Path(command[command.index('-f')+1]).read_text();compile(script,'remote','exec');seen.append(script);raise RuntimeError('local compile-only test')
 monkeypatch.setattr(backend,'run_remote_stage',probe)
 with pytest.raises(RuntimeError):backend.coordinator(cfg)
 assert len(seen)==1 and 'artifact_transfer_ready' in seen[0] and 'export_or_child' in seen[0]
