"""Verify a complete physical progress package without claiming completed research."""
from pathlib import Path
import hashlib,json,sys,xml.etree.ElementTree as ET
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
  if n in entries or Path(n).is_absolute() or '..' in Path(n).parts or len(h)!=64:raise ValueError('manifest inventory')
  entries[n]=h
 actual={p.relative_to(REPO).as_posix() for p in REPO.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}-{'MANIFEST.sha256','DELIVERY_SIGNATURE.json'}
 if actual!=set(entries):raise ValueError('physical package inventory mismatch')
 for n,h in entries.items():
  p=REPO/n
  if p.is_symlink() or digest(p)!=h:raise ValueError('physical mismatch '+n)
  with p.open('rb') as f:
   if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('unhydrated LFS '+n)
 signed=verify(json.loads((REPO/'DELIVERY_SIGNATURE.json').read_text()),pin)
 if signed['manifest_sha256']!=digest(REPO/'MANIFEST.sha256') or signed['qualification_status']!='NOT_EVALUATED' or signed['external_deployment_allowed'] is not False:raise ValueError('delivery signature/scope')
 frozen=json.loads((ROOT/'PREREGISTERED_FILES.json').read_text())
 for n,h in frozen.items():
  if digest(REPO/n)!=h:raise ValueError('frozen file changed '+n)
 status=verify(json.loads((ROOT/'signed-progress.json').read_text()),pin);report=json.loads((ROOT/'RESULTS.json').read_text())
 if status['original_results_digest']!=sha(report) or status['primary_campaign_complete']is not False or status['completed_roles']!=4 or report['campaign_complete']is not False or report['human_reviews']!=0 or report['external_deployment_allowed']is not False:raise ValueError('false completion or qualification')
 exp=ROOT/'experiment';protocol=verify(json.loads((exp/'signed_protocol.json').read_text()),pin);outcomes=json.loads((exp/'colab/outcomes.json').read_text())['runs']
 if len(outcomes)!=5 or [r['state'] for r in outcomes]!=['COMPLETED']*4+['INTERRUPTED'] or len(report['runs'])!=4:raise ValueError('failure history changed')
 screens=[]
 for i,run in enumerate(outcomes[:4]):
  bundle=exp/'artifacts'/run['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
  if sha(job)!=run['job_digest'] or sha(job)!=protocol['job_digests'][i]:raise ValueError('job substitution')
  result=validate_bundle(bundle,job,pin)
  for name,snap in result['snapshots'].items():
   for split in ('development','regression'):
    rows=[r for r in job['dataset']['records'] if r['split']==split];m=enriched(rows,snap['predictions'][split]);stored=report['runs'][i]['snapshots'][name]
    if m!=stored['metrics'][split] or public_screen(m)!=stored['screens'][split]:raise ValueError('forged model scores')
  record=json.loads((exp/'controller-replay'/(run['run']+'.json')).read_text());screens.append(verified_measurement(record,result,run['job_digest']))
 repair=ROOT/'recovery/experiment';a=verify(json.loads((repair/'signed_amendment.json').read_text()),pin)
 if a['parent_registry_export_sha256']!=digest(exp/'colab/registry-export.json') or a['parent_protocol_digest']!=sha(protocol) or a['learning_parameters_changed']is not False:raise ValueError('reset attempt accounting')
 import sqlite3
 con=sqlite3.connect('file:'+str(repair/'additional-budget.sqlite3')+'?mode=ro',uri=True);states=con.execute('SELECT state FROM attempts ORDER BY rowid').fetchall();con.close()
 if states!=[('RESERVED',)]*7:raise ValueError('unrun continuation claim false')
 owned=[exp/'colab',ROOT/'colab-tests',ROOT/'colab-tests-02',ROOT/'colab-tests-03',ROOT/'colab-tests-04',repair/'colab',ROOT/'recovery/attempts/provision-02',ROOT/'recovery/attempts/provision-03']
 for folder in owned:
  s=json.loads((folder/'monitor-state.json').read_text())
  if not s['terminal'] or not s['cleanup_verified'] or ("Session '"+s['session']+"' not found.") not in (folder/'cleanup-status.log').read_text():raise ValueError('unresolved cleanup '+str(folder))
 counts={};qa=ROOT/'colab-tests-04/evidence'
 for name,expected in [('core',605),('corpus',38)]:
  tree=ET.parse(qa/(name+'.xml')).getroot();suites=list(tree.iter('testsuite'));c={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
  if c['tests']!=expected or c['failures'] or c['errors'] or c['skipped']!=(1 if name=='core' else 0):raise ValueError('current cloud suite failure')
  if name=='core':
   skipped=[x for x in tree.iter('testcase') if x.find('skipped') is not None]
   if [(x.get('classname'),x.get('name')) for x in skipped]!=[('tests.integration.test_corrective_os_restriction','test_real_child_cannot_read_authority_secrets_or_inherit_credentials')]:raise ValueError('unexpected platform skip')
  counts[name]=c
 return {'physical_payload_files':len(entries),'frozen_files_preserved':len(frozen),'completed_t4_runs':4,'interrupted_t4_runs':1,'current_checkpoints_validated':12,'cpu_recorded_replays_verified':True,'all_cpu_decisions_match':all(s['all_decisions_match'] for s in screens),'all_cpu_probability_screens_pass':all(s['passes_registered_probability_tolerance'] for s in screens),'cpu_screens':screens,'unstarted_continuation_jobs':7,'colab_tests':counts,'owned_session_cleanups_verified':len(owned),'campaign_complete':False,'qualification_status':'NOT_EVALUATED','trust':'Controller attribution and stored CPU comparisons; no new inference or independent custody is claimed.'}
if __name__=='__main__':print(json.dumps(verify_all(sys.argv[1]),indent=2))
