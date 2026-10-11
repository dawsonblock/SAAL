import hashlib,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('/content/saal-corpus-run');root.mkdir(exist_ok=True);out=root/'evidence';out.mkdir(exist_ok=True)
def stage(name):
 print('STAGE',name,flush=True)
 with (out/'monitor.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'time':time.time()})+'\n')
report={'qualification':'NOT_EVALUATED','training_run':False,'human_reviews_completed':0,'runtime':{'python':sys.version,'platform':platform.platform()},'status':'RUNNING'}
try:
 stage('verify_inputs');inputs=json.loads(Path('/content/INPUTS.json').read_text())
 for filename,key,dest in [('corpus.zip','corpus_zip_sha256',root),('base-source.zip','base_source_zip_sha256',root/'base')]:
  p=Path('/content')/filename;assert hashlib.sha256(p.read_bytes()).hexdigest()==inputs[key]
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None
   for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
   z.extractall(dest)
 corpus=root/'SAAL-RSI-Decision-Corpus-v1-Review-Candidate';base=root/'base'
 manifest={}
 for line in (corpus/'MANIFEST.sha256').read_text().splitlines():
  sha,path=line.split('  ',1);assert hashlib.sha256((corpus/path).read_bytes()).hexdigest()==sha;manifest[path]=sha
 assert set(manifest)=={p.relative_to(corpus).as_posix() for p in corpus.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
 for path,sha in json.loads((base/'SOURCE_HASHES.json').read_text()).items():assert hashlib.sha256((base/path).read_bytes()).hexdigest()==sha
 report['input_verification']='PASS';report['inputs']=inputs
 stage('dependencies')
 p=subprocess.run([sys.executable,'-m','pip','install','pytest==8.4.2','cryptography==48.0.1'],capture_output=True,text=True,timeout=240);(out/'dependency-install.log').write_text(p.stdout+p.stderr);assert p.returncode==0
 import importlib.metadata
 report['dependencies']={n:importlib.metadata.version(n) for n in ['pytest','cryptography']};report['dependency_admission']='VERSION_PINNED_NOT_HASH_LOCKED'
 env=os.environ.copy();env['PYTHONPATH']=str(corpus/'src')+os.pathsep+str(base/'src');env['PYTHONDONTWRITEBYTECODE']='1'
 stage('full_corpus_tests');p=subprocess.run([sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider','--junitxml='+str(out/'tests.xml')],cwd=corpus,env=env,capture_output=True,text=True,timeout=240);(out/'tests.log').write_text(p.stdout+p.stderr);print(p.stdout,p.stderr,flush=True);report['test_exit_code']=p.returncode;assert p.returncode==0
 stage('rebuild_and_projection_verification');p=subprocess.run([sys.executable,'tools/verify_capsule.py'],cwd=corpus,env=env,capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr;report['capsule_verification']=json.loads(p.stdout)
 stage('review_gate_cli');p=subprocess.run([sys.executable,'-m','saal_corpus','check','corpus/CORPUS.json'],cwd=corpus,env=env,capture_output=True,text=True,timeout=60);(out/'corpus-check.log').write_text(p.stdout+p.stderr);assert p.returncode==0
 report['status']='PASS'
except Exception as e:
 report['status']='FAIL';report['error']=repr(e)
finally:
 stage('export_evidence');(out/'REPORT.json').write_text(json.dumps(report,indent=2));subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'pip-freeze.txt').open('w'),timeout=30)
 with zipfile.ZipFile('/content/saal-corpus-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.iterdir():z.write(p,p.name)
 print(json.dumps(report,indent=2),flush=True)
