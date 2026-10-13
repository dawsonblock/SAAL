"""New controller validation/transport; three remaining signed learning jobs."""
from pathlib import Path
import json,hashlib,sys,shutil
import campaign_amended as campaign
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import verify,sign
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.epoch_artifacts_v2 import validate_bundle
from saal_rsi.epoch_backend_v5 import execute_campaign
PROJECT=Path(__file__).parent


def context():
    root,pin,protocol,current,_=campaign.context()
    amendment=verify(json.loads((root/'signed_source_amendment.json').read_text()),pin)
    if amendment['parent_protocol_digest']!=sha(protocol) or amendment['current_source_map']!=current:raise ValueError('learning/source amendment mismatch')
    prior=root/'continuation-02';state=json.loads((prior/'colab/monitor-state.json').read_text())
    if not state['terminal'] or not state['cleanup_verified']:raise ValueError('previous runtime unresolved')
    ledger=EpochExperimentRegistry(prior/'additional-budget.sqlite3',pin)
    try:records=ledger.records()
    finally:ledger.close()
    if [a['digest'] for a in records['attempts']]!=[r['job_digest'] for r in protocol['runs'][4:10]] or [a['state'] for a in records['attempts']]!=['COMPLETED','COMPLETED','INTERRUPTED','RESERVED','RESERVED','RESERVED']:
        raise ValueError('only three never-started roles may be admitted')
    rows=json.loads((prior/'colab/outcomes.json').read_text())['runs']
    if len(rows)!=3 or rows[2]['error']!={'type':'ValueError','detail':'forged public metrics'}:raise ValueError('different interruption requires review')
    receipt=verify(json.loads((root/'continuation-03/signed_native_recovery.json').read_text()),pin)
    rel=Path(receipt['bundle_relative'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('recovery path')
    bundle=PROJECT/rel;job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
    result=validate_bundle(bundle,job,pin)
    if sha(job)!=protocol['runs'][6]['job_digest'] or sha(job)!=receipt['job_digest'] or sha(result)!=receipt['result_digest'] or receipt['validator_sha256']!=hashlib.sha256((campaign.ROOT/'src/saal_rsi/epoch_artifacts_v2.py').read_bytes()).hexdigest() or receipt['original_attempt_remains']!='INTERRUPTED':raise ValueError('recovered model identity')
    return root,pin,protocol,records,receipt


def sources():
    return {p:hashlib.sha256((campaign.ROOT/p).read_bytes()).hexdigest() for p in
            ('src/saal_rsi/epoch_backend_v5.py','src/saal_rsi/epoch_artifacts_v2.py','research/behavior-closure-v4/continue_after_roundoff.py')}


def prepare(key_path):
    root,pin,protocol,records,receipt=context();key=private_key(key_path)
    if key.public_key().public_bytes_raw().hex()!=pin:raise PermissionError('controller identity')
    body={'schema':'saal-public-explicit-roundoff-continuation-v1','parent_protocol_digest':sha(protocol),
          'previous_ledger_digest':sha(records),'recovery_digest':sha(receipt),'controller_source_map':sources(),
          'selected_job_digests':[r['job_digest'] for r in protocol['runs'][7:10]],
          'additional_training_reservations':3,'additional_reserved_seconds':11400,'deadline_seconds':12600,
          'full_job_digest':protocol['runs'][10]['job_digest'],'full_max_seconds':5000,
          'learning_parameters_changed':False,'automatic_training_retries':0,'probability_replay_tolerance':1e-4,
          'calibration_only_binary64_ulp_limit':8,'behavioral_rates_counts_exact':True,
          'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    with (root/'continuation-03/signed_admission.json').open('xb') as f:f.write(canonical(sign(body,key)))


def run():
    root,pin,protocol,records,receipt=context();out=root/'continuation-03'
    admission=verify(json.loads((out/'signed_admission.json').read_text()),pin)
    if admission['parent_protocol_digest']!=sha(protocol) or admission['previous_ledger_digest']!=sha(records) or admission['recovery_digest']!=sha(receipt) or admission['controller_source_map']!=sources() or admission['learning_parameters_changed'] is not False or admission['automatic_training_retries']!=0:raise ValueError('admission/source substitution')
    jobs=[json.loads((root/(r['name']+'-job.json')).read_text()) for r in protocol['runs'][7:10]]
    if [sha(verify(j,pin)) for j in jobs]!=admission['selected_job_digests']:raise ValueError('job substitution')
    execute_campaign(out/'colab',jobs,pin,root/'artifacts',out/'additional-budget.sqlite3',max_seconds=admission['deadline_seconds'])
    full=json.loads((root/(protocol['runs'][10]['name']+'-job.json')).read_text())
    if sha(verify(full,pin))!=admission['full_job_digest']:raise ValueError('full job substitution')
    free=shutil.disk_usage(PROJECT).free;required=protocol['resource_policy']['full_stage_minimum_local_free_bytes']
    if free<required:
        with (root/'full-storage-admission-rejected.json').open('xb') as f:f.write(canonical({'status':'ADMISSION_REJECTED','free_bytes':free,'required_bytes':required,'training_started':False,'qualification_status':'NOT_EVALUATED'}))
        raise RuntimeError('full recovery capacity unavailable')
    execute_campaign(root/'colab-full',[full],pin,root/'artifacts',root/'full-budget.sqlite3',max_seconds=admission['full_max_seconds'])


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run()
    else:raise ValueError('closed command')
