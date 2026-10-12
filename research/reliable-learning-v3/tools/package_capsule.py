"""Physical current-release capsule, all current tensors and full core fixtures."""
from pathlib import Path
import json,sys,zipfile,hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def package(destination,key_file):
 files={};omitted=[]
 def add(p):
  name=p.relative_to(REPO).as_posix()
  if '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix=='.pyc':return
  if p.is_symlink() or p.suffix in ('.key','.ed25519') or 'private' in p.parts:raise ValueError('private/symlink payload '+name)
  with p.open('rb') as f:
   if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('unhydrated LFS '+name)
  files[name]=p
 frozen=json.loads((ROOT/'PREREGISTERED_FILES.json').read_text())
 for n,h in frozen.items():
  if digest(REPO/n)!=h:raise ValueError('frozen source drift '+n)
  add(REPO/n)
 # Real fixtures necessary for the complete 596-case suite, not a subset count.
 qa=ROOT/('colab-tests-04' if (ROOT/'colab-tests-04/input-inventory.json').exists() else 'colab-tests-03')
 for n,h in json.loads((qa/'input-inventory.json').read_text()).items():
  if digest(REPO/n)!=h:raise ValueError('QA fixture substitution '+n)
  add(REPO/n)
 for directory in ('src','configs','docs','tests','reviewed-decision-corpus'):
  for p in (REPO/directory).rglob('*'):
   if p.is_file():add(p)
 for n in ('README.md','pyproject.toml','LICENSE','CHANGELOG.md','.gitattributes'):
  if (REPO/n).is_file():add(REPO/n)
 for p in ROOT.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if any(n.startswith('packaging') for n in rel.parts) or p.name.startswith('package-build'):continue
  if 'stage' in rel.parts or p.name in ('worker.zip','test-inputs.zip','results.zip'):omitted.append(p.relative_to(REPO).as_posix());continue
  if rel.parts[:2]==('experiment','colab'):
   tail=rel.parts[2:];keep=(len(tail)==1 and (p.suffix in ('.json','.jsonl','.log','.py','.txt','.pub') or p.name=='worker.zip')) or (len(tail)==2 and p.name=='manifest.json')
   if not keep:omitted.append(p.relative_to(REPO).as_posix());continue
  if rel.parts[0] in ('colab-tests','colab-tests-02','colab-tests-03','colab-tests-04') and (p.name=='test-inputs.zip' or p.name=='results.zip'):omitted.append(p.relative_to(REPO).as_posix());continue
  add(p)
 anchor=json.loads((ROOT/'PREREGISTRATION_ANCHOR.json').read_text());layout={'schema':'saal-public-reliable-learning-capsule-v3','source_commit':anchor['commit'],'included':'All source packages, exact jobs and protocols, all 12 recovered tensors from four T4 runs, raw results and failures, monitored lifecycles, CPU restoration evidence, CI evidence, review pilot, and physical fixtures for the complete core/corpus suites. Lost fifth-job tensors and unrun models do not exist in this package.','excluded_prerequisites':{'upstream_weights':'Standalone pinned Hugging Face snapshot/cache is not separately bundled. Full-tuning checkpoint files, when generated, necessarily contain encoder weights; this progress package has no newly completed full-tuning run.','prior_full_training_tensors':'Earlier complete v1/v2 experiment archives remain separately preserved. Exact old-case metrics are reported; revalidating full old parent checkpoints requires those archives.','protected_population':'No protected holdout or independent evaluator deployment exists in this capsule.'},'omitted_redundant_transfers':omitted,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'human_reviews':0,'campaign_complete':False,'completed_t4_runs':4,'interrupted_t4_runs':1}
 generated={'PACKAGE_LAYOUT.json':json.dumps(layout,sort_keys=True,indent=2).encode()};manifest=''.join(digest(p)+'  '+n+'\n' for n,p in sorted(files.items()))+hashlib.sha256(generated['PACKAGE_LAYOUT.json']).hexdigest()+'  PACKAGE_LAYOUT.json\n';generated['MANIFEST.sha256']=manifest.encode()
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=(ROOT/'experiment/controller.pub').read_text().strip()
 if key.public_key().public_bytes_raw().hex()!=pin:raise ValueError('signing identity')
 generated['DELIVERY_SIGNATURE.json']=canonical(sign({'schema':'saal-public-delivery-v3','manifest_sha256':hashlib.sha256(generated['MANIFEST.sha256']).hexdigest(),'preregistration_commit':anchor['commit'],'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'trust':'Controller attribution; not independent qualification.'},key))
 meta=ROOT/'packaging';meta.mkdir(exist_ok=False)
 for n,b in generated.items():(meta/n).write_bytes(b)
 with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for n,p in sorted(files.items()):z.write(p,n)
  for n,b in generated.items():z.writestr(n,b)
 info={'archive':str(destination),'bytes':Path(destination).stat().st_size,'sha256':digest(Path(destination)),'files':len(files)+len(generated),'frozen_files':len(frozen),'qualification_status':'NOT_EVALUATED'};(meta/'ARCHIVE.json').write_text(json.dumps(info,indent=2));Path(str(destination)+'.sha256').write_text(info['sha256']+'  '+Path(destination).name+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':package(Path(sys.argv[1]),sys.argv[2])
