import sys,subprocess,hashlib,json,zipfile,os,threading,time,platform
from pathlib import Path
stop=threading.Event()
def heart():
 while not stop.wait(5):print(json.dumps({'monitor':'qa_heartbeat'}),flush=True)
threading.Thread(target=heart,daemon=True).start()
root=Path('/content/saal-qa');evidence=Path('/content/saal-qa-evidence');evidence.mkdir()
try:
 if sys.version_info[:2]!=(3,13):raise ValueError('QA Python lock mismatch')
 archive=Path('/content/saal-qa-inputs.zip');lock=Path('/content/saal-qa-lock.txt')
 if hashlib.sha256(archive.read_bytes()).hexdigest()!='9672e70375c3a486c0f1dca2a424a8b95b12fe3bdce53c4fe3bf54a05e263800' or hashlib.sha256(lock.read_bytes()).hexdigest()!='fe75c6508354d719c74d50cba540d1d5306d5c74d3b77d7ebc23f1f1f34e04b4':raise ValueError('QA input substitution')
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
 for name,tests,paths in [('core','tests',str(root/'src')),('corpus','reviewed-decision-corpus/tests',str(root/'src')+':'+str(root/'reviewed-decision-corpus/src')),('reporting','research/reliable-learning-v3/tests',str(root/'src')),('project','research/behavior-closure-v4/tests',str(root/'src'))]:
  env['PYTHONPATH']=paths
  with (evidence/(name+'.log')).open('w') as log:
   returns[name]=subprocess.run([python,'-m','pytest',tests,'-q','--junitxml='+str(evidence/(name+'.xml'))],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
  print(json.dumps({'monitor':'qa_suite_complete','suite':name,'returncode':returns[name]}),flush=True)
 (evidence/'result.json').write_text(json.dumps({'returncodes':returns,'source_commit':'8d01586ef2d673fabc709dd11ca350290c62617f','input_archive_sha256':'9672e70375c3a486c0f1dca2a424a8b95b12fe3bdce53c4fe3bf54a05e263800','python':platform.python_version(),'runtime':'Colab CPU; T4 neural training is tested separately','hash_lock_enforced':True,'qualification_status':'NOT_EVALUATED'}))
 if any(returns.values()):print(json.dumps({'monitor':'worker_failure','stage':'qa','returncode':1}),flush=True)
except BaseException as exc:
 (evidence/'error.json').write_text(json.dumps({'type':type(exc).__name__,'detail':str(exc)}));print(json.dumps({'monitor':'worker_failure','stage':'qa','error':str(exc)}),flush=True)
finally:
 stop.set()
 with zipfile.ZipFile('/content/saal-qa-results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in evidence.iterdir():z.write(p,p.name)
