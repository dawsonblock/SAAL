"""Physical package, preregistration, native artifacts and delivery integrity."""
from pathlib import Path
import hashlib,json,sys
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.spec import sha
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify_all(pin):
 import torch
 torch.set_num_threads(6)
 manifest=ROOT/'MANIFEST.sha256';entries={}
 for line in manifest.read_text().splitlines():
  h,name=line.split('  ',1)
  if name in entries or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('manifest identity')
  entries[name]=h
 actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}-{'MANIFEST.sha256','DELIVERY_SIGNATURE.json'}
 if actual!=set(entries):raise ValueError('package completeness: '+str(actual^set(entries)))
 for name,h in entries.items():
  p=ROOT/name
  if p.is_symlink() or digest(p)!=h:raise ValueError('physical artifact mismatch '+name)
  with p.open('rb') as f:
   if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer '+name)
 signed=verify(json.loads((ROOT/'DELIVERY_SIGNATURE.json').read_text()),pin)
 if signed['manifest_sha256']!=digest(manifest) or signed['qualification_status']!='NOT_EVALUATED' or signed['external_deployment_allowed']is not False:raise ValueError('delivery identity/authority')
 for mapping in ['PREREGISTERED_FILES.json','SOURCE_ORIGIN.json']:
  data=json.loads((ROOT/mapping).read_text());files=data['files'] if mapping=='SOURCE_ORIGIN.json' else data
  for name,h in files.items():
   if digest(ROOT/name)!=h:raise ValueError('frozen source drift '+name)
 protocol=verify(json.loads((ROOT/'experiment/signed_protocol.json').read_text()),pin)
 outcomes=json.loads((ROOT/'experiment/colab/outcomes.json').read_text())['runs']
 if len(outcomes)!=len(protocol['runs']):raise ValueError('missing runs')
 for i,r in enumerate(outcomes):
  if r['state']!='COMPLETED':raise ValueError('incomplete run '+r['run'])
  bundle=ROOT/'experiment/artifacts'/r['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
  if sha(job)!=protocol['job_digests'][i]:raise ValueError('cross-job evidence')
  validate_bundle(bundle,job,pin)
  replay=json.loads((ROOT/'experiment/controller-replay'/(r['run']+'.json')).read_text())
  if replay['all_predictions_agree']is not True or replay['job_digest']!=sha(job):raise ValueError('missing CPU replay')
 cleanup=json.loads((ROOT/'experiment/colab/cleanup-evidence.json').read_text())
 if cleanup.get('confirmed')is not True:raise ValueError('unconfirmed Colab cleanup')
 return {'physical_files':len(entries),'runs':len(outcomes),'all_native_artifacts_validated':True,'all_three_checkpoints_cpu_replayed':True,'cleanup_evidence':cleanup,'qualification_status':'NOT_EVALUATED','trust':'Bundled pin proves self-consistency only unless independently authenticated.'}
if __name__=='__main__':print(json.dumps(verify_all(sys.argv[1] if len(sys.argv)>1 else (ROOT/'experiment/controller.pub').read_text().strip()),indent=2))
