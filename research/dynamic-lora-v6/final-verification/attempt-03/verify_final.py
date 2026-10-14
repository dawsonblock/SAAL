"""Detached final-byte verification; does not rewrite the release ZIP."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
REPO=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish')
ARCHIVE=Path('/Users/dawsonblock/Downloads/SAAL-RSI-v0.9.5-dev6-Full-Research-Evidence.zip')
CLEAN=Path('/Users/dawsonblock/Downloads/SAAL-RSI-Dynamic-LoRA-v6-clean-extraction-03')
PIN='e3db435cf14ebd609a7950c039a15809001bf00b23fa7deb363a7f1bffe0df4b'
PYTHON=sys.executable
TOOL=REPO/'tools/adaptive_capsule.py'
START=time.time()
if not sys.dont_write_bytecode: raise RuntimeError("Explicit -B is required, including when -I ignores environment settings")

def write_new(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def update_state(phase,**fields):
 value={'phase':phase,'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields}
 temporary=OUT/'verification-state.tmp';temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');temporary.replace(OUT/'verification-state.json')
def run_tool(action,*args):
 with (OUT/(action+'.log')).open('x') as log:
  r=subprocess.run([PYTHON,'-I',str(TOOL),action,*map(str,args)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 if r.returncode:raise RuntimeError(action+' failed; preserve log')
 return json.loads((OUT/(action+'.log')).read_text())
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

try:
 meta=json.loads(Path(str(ARCHIVE)+'.metadata.json').read_text())
 update_state('AUTHENTICATING_ARCHIVE')
 before=digest(ARCHIVE)
 if before!=meta['sha256'] or meta['controller_public_key']!=PIN:raise ValueError('archive identity changed')
 update_state('CLEAN_EXTRACTION',archive_sha256=before)
 extraction=run_tool('extract',ARCHIVE,CLEAN,before)
 update_state('VERIFYING_ALL_FILES',archive_sha256=before)
 verification=run_tool('verify',CLEAN,PIN)
 write_new(OUT/'extraction.json',extraction);write_new(OUT/'physical-verification.json',verification)
 locked_dirs=[CLEAN/'src']+[p for p in (CLEAN/'src').rglob('*') if p.is_dir()]
 for directory in locked_dirs: directory.chmod(directory.stat().st_mode & ~0o222)
 write_new(OUT/'readonly-source-directories.json',{'directories':[str(p.relative_to(CLEAN)) for p in locked_dirs],'scope':'Prevent isolated child-process bytecode creation; registered source bytes unchanged; output and fixtures remain writable outside src.'})
 suites=[('core','tests','src'),('corpus','reviewed-decision-corpus/tests','src:reviewed-decision-corpus/src'),('reporting','research/reliable-learning-v3/tests','src'),('project','research/behavior-closure-v4/tests','src'),('diverse','research/diverse-learning-v1/tests','src:research/diverse-learning-v1/src'),('reliability','research/learning-reliability-v2/tools/test_postprocessing.py','src'),('repairs','research/evidence-closure-v5/tests','src'),('adaptive','research/dynamic-lora-v6/tests','src')]
 results=[]
 for name,target,paths in suites:
  update_state('CANONICAL_TESTS',suite=name,finished_suites=len(results),archive_sha256=before)
  started=time.monotonic();xml=OUT/(name+'.xml')
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(str(CLEAN/p) for p in paths.split(':')))
  with (OUT/(name+'.log')).open('x') as log:
   run=subprocess.run([PYTHON,'-m','pytest','-p','no:cacheprovider',target,'-q','--junitxml='+str(xml)],cwd=CLEAN,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  root=ET.parse(xml).getroot();rows=list(root) if root.tag=='testsuites' else [root]
  counts={key:sum(int(r.attrib.get(key,0)) for r in rows) for key in ('tests','failures','errors','skipped')}
  cases=list(root.iter('testcase'))
  if len(cases)!=counts['tests'] or any(c.find('failure') is not None or c.find('error') is not None for c in cases) or counts['failures'] or counts['errors'] or run.returncode:
   raise RuntimeError('canonical suite failed: '+name)
  results.append({'suite':name,'returncode':run.returncode,**counts,'wall_seconds':time.monotonic()-started})
  print(json.dumps({'suite':name,**counts}),flush=True)
 update_state('REPRODUCING_FAILED_STUDY_CLOSURE',archive_sha256=before)
 spec=importlib.util.spec_from_file_location('closed_study',CLEAN/'research/dynamic-lora-v6/analysis/close_failed_study.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 closure=module.close_failed_study()
 if closure!=json.loads((CLEAN/'research/dynamic-lora-v6/RESULTS.json').read_text()):raise ValueError('clean closure reconstruction differs')
 from saal_rsi.model_contract import verify
 from saal_rsi.spec import sha
 signed=verify(json.loads((CLEAN/'research/dynamic-lora-v6/validation/STUDY_CLOSURE_SIGNED.json').read_text()),PIN)
 if signed['results_digest']!=sha(closure) or signed['report_sha256']!=digest(CLEAN/'research/dynamic-lora-v6/RELEASE_REPORT.md'):raise ValueError('closure attribution mismatch')
 update_state('FINAL_MEMBER_AND_ARCHIVE_RECHECK',archive_sha256=before)
 # The trusted core verifier rejects all extra files, including caches.
 spec=importlib.util.spec_from_file_location('capsule_verify',TOOL);capsule=importlib.util.module_from_spec(spec);spec.loader.exec_module(capsule)
 after_files=capsule.verify_files(CLEAN,PIN)
 after=digest(ARCHIVE)
 if after!=before:raise ValueError('tested archive bytes changed')
 total={key:sum(x[key] for x in results) for key in ('tests','failures','errors','skipped')}
 result={'schema':'saal-final-physical-release-verification-v1','status':'VERIFIED','archive':str(ARCHIVE),'archive_sha256':before,'archive_bytes':ARCHIVE.stat().st_size,'source_commit':meta['source_commit'],'evidence_commit':meta['evidence_commit'],'controller_public_key':PIN,'clean_extraction':extraction,'physical_verification_before':verification,'physical_verification_after':after_files,'canonical_suites':results,'totals':total,'passed':total['tests']-total['skipped'],'clean_closure_reconstruction':True,'controller_closure_signature_verified':True,'archive_unchanged_after_tests':True,'test_logs_outside_release_tree':True,'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED','external_deployment_allowed':False,'campaign_state':'FAILED','registered_comparison':'INCOMPLETE','elapsed_seconds':time.time()-START}
 write_new(OUT/'FINAL_VERIFICATION.json',result)
 update_state('COMPLETED',archive_sha256=before,passed=result['passed'])
 print(json.dumps({'status':'VERIFIED','archive_sha256':before,'passed':result['passed'],'totals':total}),flush=True)
except BaseException as exc:
 update_state('FAILED',error={'type':type(exc).__name__,'detail':str(exc)})
 raise
