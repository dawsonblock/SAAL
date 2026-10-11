from pathlib import Path
import os,sys,subprocess,json,time,hashlib,zipfile,traceback,platform,importlib.metadata
r=Path('/content/saal-tests');r.mkdir(exist_ok=True);start=time.time();status={}
def update(stage,**extra):
 status.update(stage=stage,elapsed_seconds=time.time()-start,**extra);(r/'status.json').write_text(json.dumps(status,indent=2));print(stage,flush=True)
def run(args,name,timeout,env=None,cwd=None):
 update(name)
 with (r/(name+'.log')).open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=env,cwd=cwd)
 if p.returncode:raise RuntimeError(name+' failed with exit '+str(p.returncode))
try:
 update('VERIFY_TRANSFER');parts=json.loads(Path('/content/parts.json').read_text());data=b''
 for part in parts['parts']:
  chunk=Path('/content',part['name']).read_bytes();assert hashlib.sha256(chunk).hexdigest()==part['sha256'];data+=chunk
 assert hashlib.sha256(data).hexdigest()==parts['sha256'];Path('/content/base-tests.zip').write_bytes(data);del data
 for archive,dest in [('/content/base-tests.zip',r/'base'),('/content/supplements.zip',r/'supplements')]:
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
   z.extractall(dest)
 assert sys.version_info[:2]==(3,13) and platform.machine()=='x86_64'
 run([sys.executable,'-m','venv',str(r/'env')],'CREATE_ENV',60);python=str(r/'env/bin/python')
 lock=r/'base/configs/dependency-locks/pretrained-linux-x86_64-py313.txt';(r/'admitted-lock.sha256').write_text(hashlib.sha256(lock.read_bytes()).hexdigest())
 run([python,'-m','pip','download','--require-hashes','--only-binary=:all:','-r',str(lock),'--dest',str(r/'wheels')],'HASHED_DOWNLOAD',900)
 (r/'wheel-inventory.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'wheels').iterdir()},indent=2))
 run([python,'-m','pip','install','--require-hashes','--no-index','--find-links',str(r/'wheels'),'-r',str(lock)],'HASHED_INSTALL',300)
 slugs=['SAAL-RSI-Semantic-Curriculum-v2','SAAL-RSI-Semantic-Grammar-v3','SAAL-RSI-Dynamic-LoRA-v4'];env=os.environ.copy();env['PYTHONPATH']=':'.join([str(r/'base/src'),str(r/'base/tests/integration'),*(str(r/'supplements'/s/'src') for s in slugs)]);env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['TOKENIZERS_PARALLELISM']='false'
 run([python,str(Path('/content/gpu_tests.py')),'--preflight',str(r/'runtime.json')],'GPU_PREFLIGHT',120,env)
 update('FULL_TESTS')
 args=[python,'-m','pytest','-o','addopts=','-q','--import-mode=importlib','--junitxml='+str(r/'tests.xml'),str(r/'base/tests'),*(str(r/'supplements'/s/'tests') for s in slugs)]
 with (r/'FULL_TESTS.log').open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=600,env=env,cwd=r/'base')
 status['test_exit_code']=p.returncode
 run([python,'-m','pytest','-o','addopts=','-q','--junitxml='+str(r/'gpu-tests.xml'),'/content/gpu_tests.py'],'GPU_TESTS',120,env,r)
 import xml.etree.ElementTree as ET
 counts={k:sum(int(s.get(k,0)) for s in ET.parse(r/'tests.xml').getroot()) for k in ['tests','failures','errors','skipped']};status['suite_counts']=counts
 if p.returncode:raise RuntimeError('Full suite failed; preserved complete log and XML')
 assert counts['tests']==542 and counts['skipped']==0
 update('COMPLETED')
except BaseException as e:
 (r/'error.txt').write_text(traceback.format_exc());update('FAILED',error=str(e))
finally:
 with zipfile.ZipFile('/content/saal-test-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in r.iterdir():
   if p.is_file():z.write(p,p.name)
