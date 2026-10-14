"""Full source tests on a separate bounded Colab CPU runtime.

Separating this runtime avoids competing with preregistered T4 training work.
Only public fixtures are uploaded; test-generated keys are ephemeral fixtures.
"""
from pathlib import Path
import hashlib,json,sys,uuid,zipfile,subprocess,os,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from saal_rsi.model_monitor_v3 import execute_colab_plan
from saal_rsi.spec import canonical,sha
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(out_name='colab-tests'):
 if out_name not in ('colab-tests','colab-tests-02','colab-tests-03','colab-tests-04'):raise ValueError('closed QA attempt identity')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 out=ROOT/out_name;out.mkdir(exist_ok=False);files={}
 def add(p):
  n=p.relative_to(REPO).as_posix()
  if '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix=='.pyc':return
  if p.is_symlink() or p.suffix in ('.key','.ed25519'):raise ValueError('unsafe QA fixture '+n)
  with p.open('rb') as f:
   if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('missing QA LFS payload '+n)
  files[n]=p
 for directory in ('src','configs','tests','tools','reviewed-decision-corpus','examples/pretrained-head-compat-v2','examples/corrective_research','examples/pretrained_research/v094-sequence-no-replay','research/reliable-learning-v3/local-smoke-03','research/diverse-learning-v1/experiment/artifacts/run-03-6616792a4767','research/reliable-learning-v3/experiment/artifacts/run-01-c6ae464acc2c','research/behavior-closure-v4/experiment/artifacts/run-02-1ead0e5752ba','research/diverse-learning-v1/src','research/diverse-learning-v1/tests','research/diverse-learning-v1/tools','research/learning-reliability-v2/tools'):
  for p in (REPO/directory).rglob('*'):
   if p.is_file():add(p)
 for n in ('pyproject.toml','provenance/v0.9.3-historical-freeze.json','examples/pretrained-controller.pub','research/reliable-learning-v3/controller.pub','research/reliable-learning-v3/experiment/controller.pub','research/reliable-learning-v3/experiment/uniform-v1_109-job.json','research/learning-reliability-v2/corpus/dataset.json','research/learning-reliability-v2/corpus/state-audit.json','research/diverse-learning-v1/experiment/diverse_dataset.json','research/diverse-learning-v1/experiment/old_control_dataset.json','research/diverse-learning-v1/experiment/controller.pub','research/behavior-closure-v4/experiment/controller.pub','research/behavior-closure-v4/dataset.json'):add(REPO/n)
 for n in json.loads((REPO/'provenance/v0.9.3-historical-freeze.json').read_text())['protected_files']:add(REPO/n)
 # Historical validation traverses the archive registry as well as direct test fixtures.
 def archives(value):
  if type(value)is dict:
   if 'archive' in value and 'archive_sha256' in value:
    name=value['archive']
    if type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('historical archive boundary')
    p=REPO/name
    if digest(p)!=value['archive_sha256']:raise ValueError('historical archive substitution')
    add(p)
   for child in value.values():archives(child)
  elif type(value)is list:
   for child in value:archives(child)
 for p in (REPO/'configs/pretrained').glob('*.json'):archives(json.loads(p.read_text()))
 # New project tests and previous reporting controls are public source fixtures.
 for directory in ('research/behavior-closure-v4','research/reliable-learning-v3/tools','research/reliable-learning-v3/tests','research/evidence-closure-v5'):
  for p in (REPO/directory).rglob('*.py'):
   if not any(x.startswith('colab-tests') or x in ('stage','packaging-attempt-01','colab-paired','colab-full','cpu-replay') for x in p.parts):add(p)
 for p in (REPO/'research/diverse-learning-v1/experiment').glob('*-job.json'):add(p)
 add(REPO/'research/diverse-learning-v1/experiment/signed_protocol.json')
 for directory in ('research/dynamic-lora-v6/tests','research/dynamic-lora-v6/native-engineering-05'):
  for p in (REPO/directory).rglob('*'):
   if p.is_file() and 'worker' not in p.relative_to(REPO/directory).parts:add(p)
 # Portability tests import the controller entry point. Include its real source
 # rather than relying on files available only in the developer checkout.
 for p in ROOT.glob('*.py'):add(p)
 for n in ('FOUNDATION.json','controller.pub','README.md'):add(ROOT/n)
 sys.path.insert(0,str(REPO/'tools'))
 from adaptive_qa_identity import source_identity
 identity=source_identity(REPO,files,commit)
 (out/'source-identity.json').write_bytes(canonical(identity))
 inventory={n:digest(p) for n,p in sorted(files.items())};(out/'input-inventory.json').write_bytes(canonical(inventory))
 archive=out/'test-inputs.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n,p in sorted(files.items()):z.write(p,n)
  z.writestr('QA_INPUT_INVENTORY.json',canonical(inventory))
 # Verify this exact minimal payload, including transitive history archives,
 # before allocating another runtime. Preserve preflight failure logs.
 with tempfile.TemporaryDirectory(prefix='saal-qa-preflight-') as temp:
  extracted=Path(temp)
  with zipfile.ZipFile(archive) as z:z.extractall(extracted)
  for name,h in inventory.items():
   if digest(extracted/name)!=h:raise ValueError('clean QA input substitution '+name)
  for name,tests,paths in [('core','tests',str(extracted/'src')),('corpus','reviewed-decision-corpus/tests',str(extracted/'src')+':'+str(extracted/'reviewed-decision-corpus/src')),('reporting','research/reliable-learning-v3/tests',str(extracted/'src')),('project','research/behavior-closure-v4/tests',str(extracted/'src')),('diverse','research/diverse-learning-v1/tests',str(extracted/'src')+':'+str(extracted/'research/diverse-learning-v1/src')),('reliability','research/learning-reliability-v2/tools/test_postprocessing.py',str(extracted/'src')),('repairs','research/evidence-closure-v5/tests',str(extracted/'src')),('adaptive','research/dynamic-lora-v6/tests',str(extracted/'src'))]:
   with (out/('local-'+name+'.log')).open('w') as log:
    subprocess.run([sys.executable,'-m','pytest',tests,'-q','--junitxml='+str(out/('local-'+name+'.xml'))],cwd=extracted,env=dict(os.environ,PYTHONPATH=paths),stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
 session='saal-'+uuid.uuid4().hex[:20];lock=REPO/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt'
 code="""import sys,subprocess,hashlib,json,zipfile,os,threading,time,platform
from pathlib import Path
stop=threading.Event()
def heart():
 while not stop.wait(5):print(json.dumps({'monitor':'qa_heartbeat'}),flush=True)
threading.Thread(target=heart,daemon=True).start()
root=Path('/content/saal-qa');evidence=Path('/content/saal-qa-evidence');evidence.mkdir()
try:
 if sys.version_info[:2]!=(3,13):raise ValueError('QA Python lock mismatch')
 archive=Path('/content/saal-qa-inputs.zip');lock=Path('/content/saal-qa-lock.txt')
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_DIGEST or hashlib.sha256(lock.read_bytes()).hexdigest()!=LOCK_DIGEST:raise ValueError('QA input substitution')
 with zipfile.ZipFile(archive) as z:
  if len(z.namelist())!=len(set(z.namelist())) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('QA archive bounds')
  z.extractall(root)
 for n,h in json.loads((root/'QA_INPUT_INVENTORY.json').read_text()).items():
  with (root/n).open('rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise ValueError('QA file substitution')
 subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-qa-env'],check=True,timeout=30)
 python='/content/saal-qa-env/bin/python'
 subprocess.run(['uv','pip','install','--python',python,'--require-hashes','-r',str(lock)],check=True,timeout=900)
 subprocess.run(['node','--version'],check=True,timeout=10,stdout=(evidence/'node-version.txt').open('w'))
 env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
 returns={}
 for name,tests,paths in [('core','tests',str(root/'src')),('corpus','reviewed-decision-corpus/tests',str(root/'src')+':'+str(root/'reviewed-decision-corpus/src')),('reporting','research/reliable-learning-v3/tests',str(root/'src')),('project','research/behavior-closure-v4/tests',str(root/'src')),('diverse','research/diverse-learning-v1/tests',str(root/'src')+':'+str(root/'research/diverse-learning-v1/src')),('reliability','research/learning-reliability-v2/tools/test_postprocessing.py',str(root/'src')),('repairs','research/evidence-closure-v5/tests',str(root/'src')),('adaptive','research/dynamic-lora-v6/tests',str(root/'src'))]:
  env['PYTHONPATH']=paths
  with (evidence/(name+'.log')).open('w') as log:
   returns[name]=subprocess.run([python,'-m','pytest',tests,'-q','--junitxml='+str(evidence/(name+'.xml'))],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
  print(json.dumps({'monitor':'qa_suite_complete','suite':name,'returncode':returns[name]}),flush=True)
 (evidence/'result.json').write_text(json.dumps({'returncodes':returns,'source_commit':SOURCE_COMMIT,'input_archive_sha256':ARCHIVE_DIGEST,'python':platform.python_version(),'runtime':'Colab CPU; T4 neural training is tested separately','hash_lock_enforced':True,'qualification_status':'NOT_EVALUATED'}))
 if any(returns.values()):print(json.dumps({'monitor':'worker_failure','stage':'qa','returncode':1}),flush=True)
except BaseException as exc:
 (evidence/'error.json').write_text(json.dumps({'type':type(exc).__name__,'detail':str(exc)}));print(json.dumps({'monitor':'worker_failure','stage':'qa','error':str(exc)}),flush=True)
finally:
 stop.set()
 with zipfile.ZipFile('/content/saal-qa-results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in evidence.iterdir():z.write(p,p.name)
"""
 # Fixed-content upload chunks are idempotent. A failed large HTTP upload is
 # preserved in prior QA06; retries here apply only to these exact public bytes.
 parts=[]
 with archive.open('rb') as source:
  i=0
  while chunk:=source.read(32_000_000):
   name=f'qa-input-part-{i:03d}.bin';part=out/name;part.write_bytes(chunk)
   parts.append({'name':name,'size':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()});i+=1
 assemble="""import hashlib,json
from pathlib import Path
parts=PARTS
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!=WHOLE:raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
""".replace('PARTS',repr(parts)).replace('WHOLE',repr(digest(archive)))
 assembly=out/'assemble.py';assembly.write_text(assemble)
 upload=out/'upload_chunks.py'
 uploader="""import sys,time,json
from pathlib import Path
sys.path.insert(0,SOURCE_ROOT)
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path(WORKSPACE);session=SESSION
for item in PARTS:
 path=workspace/item['name']
 if path.stat().st_size!=item['size'] or file_hash(path)!=item['sha256']:raise ValueError('local QA chunk changed')
 for attempt in range(1,6):
  try:
   stream_command(['colab','upload','-s',session,str(path),'/content/'+item['name']],workspace/(item['name']+f'-upload-{attempt}.log'),90)
   break
  except (RuntimeError,TimeoutError):
   if attempt==5:raise
   time.sleep(min(10,2**attempt))
 print(json.dumps({'monitor':'qa_chunk_uploaded','name':item['name'],'bytes':item['size']}),flush=True)
""".replace('SOURCE_ROOT',repr(str(REPO/'src'))).replace('WORKSPACE',repr(str(out.resolve()))).replace('SESSION',repr(session)).replace('PARTS',repr(parts))
 upload.write_text(uploader)
 code=code.replace('ARCHIVE_DIGEST',repr(digest(archive))).replace('LOCK_DIGEST',repr(digest(lock))).replace('SOURCE_COMMIT',repr(commit));script=out/'run.py';script.write_text(code)
 commands=[['colab','new','-s',session],[sys.executable,'-I',str(upload)],['colab','exec','-s',session,'-f',str(assembly),'--timeout','300'],['colab','upload','-s',session,str(lock),'/content/saal-qa-lock.txt'],['colab','exec','-s',session,'-f',str(script),'--timeout','3400'],['colab','download','-s',session,'/content/saal-qa-results.zip',str(out/'results.zip')],['colab','stop','-s',session]]
 plan={'schema':'saal-public-colab-tests-v1','session':session,'job_digest':sha({'inventory':inventory,'script':digest(script),'commit':commit}),'estimated_max_seconds':7200,'commands':commands,'phases':['PROVISIONING','UPLOAD_INPUT_CHUNKS','ASSEMBLE_INPUTS','UPLOAD_LOCK','TESTING','RECOVERY'],'external_deployment_allowed':False}
 (out/'plan.json').write_bytes(canonical(plan));state=execute_colab_plan(plan,out)
 with zipfile.ZipFile(out/'results.zip') as z:
  if len(z.infolist())>20 or sum(i.file_size for i in z.infolist())>50_000_000 or any('/' in n or '..' in n for n in z.namelist()):raise ValueError('QA recovery bounds')
  z.extractall(out/'evidence')
 # Validate all eight suites, their actual JUnit cases, source/input identity,
 # and confirmed teardown. The inherited seven-suite check omitted adaptive.
 sys.path.insert(0,str(REPO/'tools'))
 from adaptive_qa_result import validate_results
 result=validate_results(out/'evidence',state,commit,digest(archive))
 (out/'verified-results.json').write_bytes(canonical(result))
 print(json.dumps(result));return result
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'colab-tests')
