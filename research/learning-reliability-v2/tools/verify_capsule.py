"""Verify physical release bytes and reproduce native candidate admission checks."""
from pathlib import Path
import hashlib, json, sys
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.spec import sha
from saal_rsi.learning_audit import audit_bundle
from saal_rsi.sequence_replay_v2 import prediction_equal

ROOT=Path(__file__).resolve().parents[1]
CAPSULE=ROOT.parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_recorded_replay(replay,result,job_digest):
    """A faithfully recorded numerical failure is evidence, not a missing file."""
    if replay['job_digest']!=job_digest or replay['probability_tolerance']!=1e-4:raise ValueError('replay identity/tolerance')
    seen=set();agrees=[];labels=[];deltas=[]
    for m in replay['measurements']:
        pair=(m['checkpoint'],m['split'])
        if pair in seen or pair not in {(c,s) for c in ('initial','best','terminal') for s in ('development','regression')}:raise ValueError('replay measurement substitution')
        seen.add(pair);expected=result['snapshots'][pair[0]]['predictions'][pair[1]];actual=m['predictions']
        if len(actual)!=m['records'] or len(actual)!=len(expected) or not prediction_equal(actual,actual,'modernbert-seqcls-v1',1e-4):raise ValueError('malformed recorded CPU predictions')
        agreement=prediction_equal(actual,expected,'modernbert-seqcls-v1',1e-4)
        delta=max(abs(a-b) for x,y in zip(actual,expected) for a,b in zip(x['probabilities'],y['probabilities']))
        if agreement!=m['prediction_agreement'] or delta!=m['max_probability_difference']:raise ValueError('forged replay outcome')
        agrees.append(agreement);deltas.append(delta);labels.append(all(a['decision']==b['decision'] for a,b in zip(actual,expected)))
    if len(seen)!=6 or all(agrees)!=replay['all_predictions_agree']:raise ValueError('incomplete/forged aggregate replay')
    return {'passes_registered_probability_tolerance':all(agrees),'all_decisions_match':all(labels),'max_probability_difference':max(deltas)}

def verify_all(pin):
    import torch
    torch.set_num_threads(6)
    manifest=CAPSULE/'MANIFEST.sha256';entries={}
    for line in manifest.read_text().splitlines():
        h,name=line.split('  ',1)
        if name in entries or Path(name).is_absolute() or '..' in Path(name).parts or len(h)!=64:raise ValueError('manifest identity')
        entries[name]=h
    actual={p.relative_to(CAPSULE).as_posix() for p in CAPSULE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}-{'MANIFEST.sha256','DELIVERY_SIGNATURE.json'}
    if actual!=set(entries):raise ValueError('package inventory '+str(actual^set(entries)))
    for name,h in entries.items():
        p=CAPSULE/name
        if p.is_symlink() or digest(p)!=h:raise ValueError('physical mismatch '+name)
        with p.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer '+name)
    signed=verify(json.loads((CAPSULE/'DELIVERY_SIGNATURE.json').read_text()),pin)
    if signed['manifest_sha256']!=digest(manifest) or signed['qualification_status']!='NOT_EVALUATED' or signed['external_deployment_allowed']is not False:raise ValueError('delivery authority/identity')
    frozen=json.loads((ROOT/'PREREGISTERED_FILES.json').read_text())
    for name,h in frozen.items():
        if digest(CAPSULE/name)!=h:raise ValueError('preregistered drift '+name)
    anchor=json.loads((ROOT/'PREREGISTRATION_ANCHOR.json').read_text())
    if anchor['frozen_file_map_sha256']!=digest(ROOT/'PREREGISTERED_FILES.json') or signed['preregistration_commit']!=anchor['commit']:raise ValueError('anchor substitution')
    protocol=verify(json.loads((ROOT/'experiment/signed_protocol.json').read_text()),pin)
    outcomes=json.loads((ROOT/'experiment/colab/outcomes.json').read_text())['runs']
    if len(outcomes)!=len(protocol['runs']):raise ValueError('missing attempts')
    report=json.loads((ROOT/'RESULTS.json').read_text())
    if report['qualification_status']!='NOT_EVALUATED' or report['external_deployment_allowed'] is not False or report['human_reviews']!=0:raise ValueError('unsupported qualification/review claim')
    replay_screens=[]
    for i,run in enumerate(outcomes):
        if run['state']!='COMPLETED':raise ValueError('interrupted run retained '+run['run'])
        bundle=ROOT/'experiment/artifacts'/run['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
        if sha(job)!=protocol['job_digests'][i] or sha(job)!=run['job_digest']:raise ValueError('cross-job bundle')
        result=validate_bundle(bundle,job,pin)
        measured=audit_bundle(bundle,pin)
        if measured!=report['runs'][i]['learning_audit']:raise ValueError('public learning report substitution')
        replay=json.loads((ROOT/'experiment/controller-replay'/(run['run']+'.json')).read_text())
        replay_screens.append(verify_recorded_replay(replay,result,sha(job)))
    cleanup=json.loads((ROOT/'experiment/colab/cleanup-evidence.json').read_text())
    if cleanup['confirmed'] is not True or cleanup['confirmation_sha256']!=digest(ROOT/'experiment/colab/cleanup-status.log'):raise ValueError('unconfirmed cleanup')
    return {'physical_payload_files':len(entries),'preregistered_files_preserved':len(frozen),'runs':len(outcomes),'candidate_checkpoints':len(outcomes)*3,'native_candidates_validated':True,'recorded_cpu_replay_verified':True,'cpu_reproduction_screens':replay_screens,'all_cpu_probability_screens_pass':all(s['passes_registered_probability_tolerance'] for s in replay_screens),'cleanup_evidence_verified':True,'qualification_status':'NOT_EVALUATED','limits':'Signature pin establishes controller attribution only. Native artifact validation is rerun here; stored CPU evidence is verified, not a new CPU inference run. Faithfully recorded failures stay failures. Prior full reference checkpoints and upstream pretrained weights are separate authenticated prerequisites.'}
if __name__=='__main__':print(json.dumps(verify_all(sys.argv[1] if len(sys.argv)>1 else (ROOT/'experiment/controller.pub').read_text().strip()),indent=2))
