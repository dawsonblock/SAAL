"""Authenticate physical delivery and recompute current candidate evidence."""
from pathlib import Path
import hashlib,json,sys
from saal_rsi.model_contract import verify
from saal_rsi.spec import sha
from saal_rsi.epoch_artifacts import validate_bundle
from saal_rsi.learning_audit import public_screen
from replay_results import verified_measurement
from report_results import enriched
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_all(pin):
 entries={}
 for line in (REPO/'MANIFEST.sha256').read_text().splitlines():
  h,n=line.split('  ',1)
  if n in entries or Path(n).is_absolute() or '..' in Path(n).parts or len(h)!=64:raise ValueError('manifest path inventory')
  entries[n]=h
 actual={p.relative_to(REPO).as_posix() for p in REPO.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}-{'MANIFEST.sha256','DELIVERY_SIGNATURE.json'}
 if actual!=set(entries):raise ValueError('physical inventory mismatch')
 for n,h in entries.items():
  p=REPO/n
  if p.is_symlink() or digest(p)!=h:raise ValueError('physical substitution '+n)
  with p.open('rb') as f:
   if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('missing LFS bytes '+n)
 signature=verify(json.loads((REPO/'DELIVERY_SIGNATURE.json').read_text()),pin);anchor=json.loads((ROOT/'PREREGISTRATION_ANCHOR.json').read_text())
 if signature['manifest_sha256']!=digest(REPO/'MANIFEST.sha256') or signature['preregistration_commit']!=anchor['commit'] or signature['qualification_status']!='NOT_EVALUATED' or signature['external_deployment_allowed'] is not False:raise ValueError('delivery attribution/authority')
 frozen=json.loads((ROOT/'PREREGISTERED_FILES.json').read_text())
 if anchor['frozen_manifest_sha256']!=digest(ROOT/'PREREGISTERED_FILES.json'):raise ValueError('source anchor')
 for n,h in frozen.items():
  if digest(REPO/n)!=h:raise ValueError('frozen source drift '+n)
 exp=ROOT/'experiment';protocol=verify(json.loads((exp/'signed_protocol.json').read_text()),pin);outcomes=json.loads((exp/'colab/outcomes.json').read_text())['runs'];report=json.loads((ROOT/'RESULTS.json').read_text())
 if len(outcomes)!=len(protocol['runs']) or len(report['runs'])!=len(outcomes):raise ValueError('missing attempt evidence')
 if report['human_reviews']!=0 or report['qualification_status']!='NOT_EVALUATED' or report['external_deployment_allowed'] is not False:raise ValueError('unsupported claim')
 screens=[];eligible=[]
 for i,run in enumerate(outcomes):
  if run['job_digest']!=protocol['job_digests'][i] or run['state']!='COMPLETED':raise ValueError('incomplete/cross-job outcome')
  bundle=exp/'artifacts'/run['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
  if sha(job)!=run['job_digest']:raise ValueError('job substitution')
  result=validate_bundle(bundle,job,pin)
  for name,snap in result['snapshots'].items():
   for split in ('development','regression'):
    rows=[r for r in job['dataset']['records'] if r['split']==split];m=enriched(rows,snap['predictions'][split]);stored=report['runs'][i]['snapshots'][name]
    if m!=stored['metrics'][split] or public_screen(m)!=stored['screens'][split]:raise ValueError('forged public score')
    if split=='development' and snap['step']>0 and public_screen(m)['passes_public_screen'] and m['family_macro_accuracy']>result['snapshots']['initial']['metrics']['development']['family_macro_accuracy']:eligible.append((run['run'],name))
  record=json.loads((exp/'controller-replay'/(run['run']+'.json')).read_text());screens.append(verified_measurement(record,result,run['job_digest']))
 sequential=report['sequential_phase']['status']
 if (eligible and sequential!='ELIGIBLE_REQUIRES_NEW_PREREGISTRATION') or (not eligible and sequential!='SKIPPED_NOT_ELIGIBLE'):raise ValueError('conditional phase misreported')
 for d in (exp/'colab',ROOT/'colab-tests',ROOT/'colab-tests-02',ROOT/'colab-tests-03'):
  state=json.loads((d/'monitor-state.json').read_text())
  if not state['terminal'] or not state['cleanup_verified'] or ("Session '"+state['session']+"' not found.") not in (d/'cleanup-status.log').read_text():raise ValueError('unresolved runtime cleanup')
 import xml.etree.ElementTree as ET
 qa=ROOT/'colab-tests-03/evidence';counts={}
 for name,n in [('core',596),('corpus',38)]:
  suites=list(ET.parse(qa/(name+'.xml')).getroot().iter('testsuite'));t={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
  if t['tests']!=n or t['failures'] or t['errors'] or t['skipped']!=(1 if name=='core' else 0):raise ValueError('Colab full regression failure or unexpected skips')
  if name=='core':
   skipped=[case for case in ET.parse(qa/(name+'.xml')).getroot().iter('testcase') if case.find('skipped') is not None]
   if [(c.get('classname'),c.get('name')) for c in skipped]!=[('tests.integration.test_corrective_os_restriction','test_real_child_cannot_read_authority_secrets_or_inherit_credentials')]:raise ValueError('substituted platform skip')
  counts[name]=t
 return {'physical_payload_files':len(entries),'frozen_files_preserved':len(frozen),'runs':len(outcomes),'checkpoints':len(outcomes)*3,'native_candidates_revalidated':True,'recorded_cpu_inference_verified':True,'all_cpu_probability_screens_pass':all(s['passes_registered_probability_tolerance'] for s in screens),'all_cpu_decisions_match':all(s['all_decisions_match'] for s in screens),'cpu_screens':screens,'colab_tests':counts,'all_four_runtime_cleanups_confirmed':True,'earlier_qa_attempts':'13 then 1 missing historical archive fixture failures retained; complete closure preflight and third cloud attempt follow.','sequential_phase':sequential,'qualification_status':'NOT_EVALUATED','trust':'Controller signatures and recorded evidence, not operationally independent custody. This verifier does not rerun inference; use replay_results.py for new inference.'}
if __name__=='__main__':print(json.dumps(verify_all(sys.argv[1]),indent=2))
