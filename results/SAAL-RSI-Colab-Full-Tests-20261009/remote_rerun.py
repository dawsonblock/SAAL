from pathlib import Path
import subprocess,os,json,shutil,time,zipfile,traceback,hashlib,xml.etree.ElementTree as ET
r=Path('/content/saal-tests-r3');r.mkdir(exist_ok=True);prior=Path('/content/saal-tests-r2');start=time.time();status={}
def stage(name):status.update(stage=name,elapsed_seconds=time.time()-start);(r/'status.json').write_text(json.dumps(status,indent=2))
def run(args,name,timeout,env):
 stage(name)
 with (r/(name+'.log')).open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=env,cwd=r/'base')
 return p.returncode
try:
 stage('COPY_UNCHANGED_BASE');shutil.copytree(prior/'base',r/'base');patch=Path('/content/test_attempt_mtls_process_py313.py');target=r/'base/tests/integration/test_attempt_mtls_process.py';original=hashlib.sha256(target.read_bytes()).hexdigest();shutil.copyfile(patch,target);(r/'source-delta.json').write_text(json.dumps({'changed_test':'tests/integration/test_attempt_mtls_process.py','original_sha256':original,'new_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'production_source_changes':0,'reason':'Strict RFC 5280 CA/leaf certificate extensions for Python 3.13'},indent=2))
 python=str(prior/'env/bin/python');slugs=['SAAL-RSI-Semantic-Curriculum-v2','SAAL-RSI-Semantic-Grammar-v3','SAAL-RSI-Dynamic-LoRA-v4'];env=os.environ.copy();env['PYTHONPATH']=':'.join([str(r/'base/src'),str(r/'base/tests/integration'),*(str(prior/'supplements'/s/'src') for s in slugs)]);env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['TOKENIZERS_PARALLELISM']='false'
 args=[python,'-m','pytest','-o','addopts=','-q','--import-mode=importlib']
 assert run(args+[str(target),'/content/gpu_tests-r3.py','--junitxml='+str(r/'targeted.xml')],'TARGETED_REPAIRS',120,env)==0
 assert run(args+[str(r/'base/tests'),*(str(prior/'supplements'/s/'tests') for s in slugs),'--junitxml='+str(r/'tests.xml')],'FULL_TESTS',600,env)==0
 assert run(args+['/content/gpu_tests-r3.py','--junitxml='+str(r/'gpu-tests.xml')],'GPU_TESTS',120,env)==0
 counts={k:sum(int(s.get(k,0)) for s in ET.parse(r/'tests.xml').getroot()) for k in ['tests','failures','errors','skipped']};assert counts=={'tests':542,'failures':0,'errors':0,'skipped':1};status['suite_counts']=counts
 stage('COMPLETED_WITH_PLATFORM_SKIP')
except BaseException as e:(r/'error.txt').write_text(traceback.format_exc());status['error']=str(e);stage('FAILED')
finally:
 with zipfile.ZipFile('/content/saal-rerun-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in r.iterdir():
   if p.is_file():z.write(p,p.name)
