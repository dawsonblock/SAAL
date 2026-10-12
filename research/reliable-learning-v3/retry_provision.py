"""Fresh runtime lifecycle for already reserved, never-started signed jobs."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.epoch_backend_v2 import setup_plan
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.model_contract import verify,sign
from saal_rsi.model_monitor import write_json,execute_colab_plan
from saal_rsi.spec import sha,canonical
PROJECT=Path(__file__).parent;EXP=PROJECT/'recovery/experiment'
def run(key_file,number):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 if number not in (2,3,4):raise ValueError('bounded provisioning attempts')
 pin=(EXP/'controller.pub').read_text().strip();amend=verify(json.loads((EXP/'signed_amendment.json').read_text()),pin);ledger=EXP/'additional-budget.sqlite3';reg=EpochExperimentRegistry(ledger,pin);records=reg.records();reg.close()
 expected=[r['job_digest'] for r in amend['roles']]
 if [r['digest'] for r in records['attempts']]!=expected or any(r['state']!='RESERVED' for r in records['attempts']):raise ValueError('existing reservation consumed; provisioning retry cannot restart training')
 previous=EXP/'colab' if number==2 else PROJECT/f'recovery/attempts/provision-{number-1:02d}'
 state=json.loads((previous/'monitor-state.json').read_text())
 if not state['terminal'] or not state['cleanup_verified'] or state.get('completed_runs') is not None:raise ValueError('prior runtime not absent or inference was started')
 out=PROJECT/f'recovery/attempts/provision-{number:02d}';out.parent.mkdir(exist_ok=True);plan=setup_plan(out,28000)
 paths=[]
 for role in amend['roles']:
  p=EXP/(role['name']+'-job.json')
  if sha(verify(json.loads(p.read_text()),pin))!=role['job_digest']:raise ValueError('reserved job substitution')
  paths.append(str(p.resolve()))
 artifacts=EXP/'artifacts';artifacts.mkdir(exist_ok=True)
 config={'workspace':str(out.resolve()),'session':plan['session'],'pin':pin,'jobs':paths,'artifacts':str(artifacts.resolve()),'ledger':str(ledger.resolve())};write_json(out/'coordinator.json',config)
 launcher=out/'coordinator.py';launcher.write_text("import sys\nsys.path.insert(0,"+repr(str(ROOT/'src'))+")\nfrom saal_rsi.epoch_backend_v2 import coordinator\ncoordinator("+repr(str((out/'coordinator.json').resolve()))+")\n")
 plan['commands']+=[[sys.executable,'-I',str(launcher)],['colab','stop','-s',plan['session']]];plan['phases']+=['CAMPAIGN'];write_json(out/'submission-plan.json',plan)
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes())
 if key.public_key().public_bytes_raw().hex()!=pin:raise ValueError('controller identity')
 pointer={'schema':'saal-reserved-execution-v1','relative_workspace':out.relative_to(PROJECT).as_posix(),'session':plan['session'],'amendment_digest':sha(amend),'all_prior_jobs_unstarted':True,'new_training_budget':False,'provisioning_attempt':number,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'transport_sha256':hashlib.sha256((ROOT/'src/saal_rsi/epoch_backend_v2.py').read_bytes()).hexdigest(),'cli_version':'0.7.4','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
 (out/'signed_execution.json').write_bytes(canonical(sign(pointer,key)));(EXP/'signed_execution_location.json').write_bytes(canonical(sign(pointer,key)))
 try:return execute_colab_plan(plan,out)
 finally:
  status=json.loads((out/'monitor-state.json').read_text());write_json(out/'cleanup-evidence.json',{'session':plan['session'],'confirmed':status['cleanup_verified'],'confirmation_sha256':hashlib.sha256((out/'cleanup-status.log').read_bytes()).hexdigest() if (out/'cleanup-status.log').exists() else None})
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
