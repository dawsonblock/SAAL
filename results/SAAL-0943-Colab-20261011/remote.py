from pathlib import Path
import subprocess,sys,os,json,zipfile,hashlib,platform,time,traceback,xml.etree.ElementTree as ET
root=Path('/content/saal-0943');root.mkdir(exist_ok=True);e=root/'evidence';e.mkdir(exist_ok=True);start=time.time();report={'schema':'saal-0943-linux-validation-v1','status':'RUNNING','qualification_status':'NOT_EVALUATED','training_experiment':False}
def stage(name):
 report['stage']=name;report['elapsed_seconds']=time.time()-start;(e/'STATUS.json').write_text(json.dumps(report,indent=2));print(name,flush=True)
 with (e/'monitor.jsonl').open('a') as f:f.write(json.dumps({'stage':name,'elapsed_seconds':report['elapsed_seconds']})+'\n')
def run(args,name,timeout,env=None,cwd=None):
 stage(name)
 with (e/(name+'.log')).open('w') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=env,cwd=cwd)
 assert p.returncode==0,name+' exit '+str(p.returncode)
try:
 stage('VERIFY_TRANSFER');parts=json.loads(Path('/content/parts.json').read_text());archive=root/'input.zip'
 with archive.open('wb') as out:
  for part in parts['parts']:
   data=Path('/content',part['name']).read_bytes();assert hashlib.sha256(data).hexdigest()==part['sha256'];out.write(data)
 with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==parts['sha256']
 base=root/'base'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
  z.extractall(base)
 inventory=json.loads((base/'TEST_INPUT_HASHES.json').read_text())
 for name,sha in inventory.items():
  with (base/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
 report['input_zip_sha256']=parts['sha256'];report['source_files_verified']=len(inventory)
 supplemental=Path('/content/build_reproducible_release.py');assert hashlib.sha256(supplemental.read_bytes()).hexdigest()=='13a631f1435ebef91e52356268452a0acce3ac8d7d3c8f4f20a72a2450d1e05f'
 (base/'tools/build_reproducible_release.py').write_bytes(supplemental.read_bytes());report['supplemental_builder_sha256']='13a631f1435ebef91e52356268452a0acce3ac8d7d3c8f4f20a72a2450d1e05f'
 assert sys.version_info[:2]==(3,13) and platform.machine()=='x86_64'
 report['runtime']={'python':sys.version,'platform':platform.platform()}
 run([sys.executable,'-m','venv','--without-pip',str(root/'env')],'CREATE_ENV',60);python=str(root/'env/bin/python');lock=base/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt';report['lock_sha256']=hashlib.sha256(lock.read_bytes()).hexdigest()
 run([sys.executable,'-m','pip','--python',python,'download','--require-hashes','--only-binary=:all:','-r',str(lock),'--dest',str(root/'wheels')],'HASH_LOCKED_DOWNLOAD',900)
 (e/'wheel-inventory.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'wheels').iterdir()},indent=2))
 run([sys.executable,'-m','pip','--python',python,'install','--require-hashes','--no-index','--find-links',str(root/'wheels'),'-r',str(lock)],'HASH_LOCKED_INSTALL',300)
 env=os.environ.copy();env.update(PYTHONPATH=str(base/'src')+':'+str(base/'reviewed-decision-corpus/src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 run([python,'-m','pip','freeze'],'ENVIRONMENT_FREEZE',60,env)
 run([python,'-m','pytest','-o','addopts=','-q','--import-mode=importlib','--junitxml='+str(e/'tests.xml'),str(base/'tests'),str(base/'reviewed-decision-corpus/tests')],'FULL_TESTS',600,env,base)
 counts={k:sum(int(x.get(k,0)) for x in ET.parse(e/'tests.xml').getroot()) for k in ['tests','failures','errors','skipped']};report['counts']=counts;assert counts['tests']==544 and counts['failures']==counts['errors']==0 and counts['skipped']==1
 run([python,str(base/'reviewed-decision-corpus/tools/verify_capsule.py')],'CORPUS_REBUILD',60,env,base)
 gpu=root/'gpu_probe.py';gpu.write_text("import torch,json,sys\nfrom safetensors.torch import save_file,load_file\nassert torch.cuda.is_available()\nassert 'T4' in torch.cuda.get_device_name(0)\ntorch.manual_seed(9);m=torch.nn.Linear(4,2).cuda();x=torch.randn(8,4,device='cuda');o=torch.optim.AdamW(m.parameters());before=m.weight.detach().clone();loss=m(x).square().mean();loss.backward();o.step();assert not torch.equal(before,m.weight);torch.cuda.synchronize();save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},'/content/probe.safetensors');print(json.dumps({'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'finite_loss':bool(torch.isfinite(loss))}))\n")
 run([python,str(gpu)],'CUDA_FORWARD_BACKWARD',120,env)
 run([python,'-c',"import torch;from safetensors.torch import load_file;m=torch.nn.Linear(4,2);m.load_state_dict(load_file('/content/probe.safetensors'));assert torch.isfinite(m(torch.ones(1,4))).all();print('fresh-process CPU restore PASS')"],'FRESH_PROCESS_PROBE_REPLAY',120,env)
 report['status']='PASS';report['limits']='T4 numerical probe and software tests only; no pretrained learning campaign or independent qualification.';stage('COMPLETE')
except BaseException as ex:
 report['status']='FAIL';report['error']=str(ex);(e/'ERROR.txt').write_text(traceback.format_exc());stage('FAILED')
finally:
 (e/'REPORT.json').write_text(json.dumps(report,indent=2))
 with zipfile.ZipFile('/content/saal-0943-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in e.iterdir():z.write(p,p.name)
 print(json.dumps(report,indent=2),flush=True)
