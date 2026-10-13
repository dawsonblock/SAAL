"""Preserve the interrupted transport, admit only six unstarted paired roles."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil
import campaign_amended as campaign
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.epoch_artifacts import validate_bundle
from saal_rsi.epoch_backend_v4 import execute_campaign

PROJECT=Path(__file__).parent


def context():
    root,pin,protocol,current,changed=campaign.context()
    prior=root/'continuation-01';state=json.loads((prior/'colab/monitor-state.json').read_text())
    if not state['terminal'] or not state['cleanup_verified']:raise ValueError('previous runtime unresolved')
    amendment=verify(json.loads((root/'signed_source_amendment.json').read_text()),pin)
    if amendment['current_source_map']!=current or amendment['parent_protocol_digest']!=sha(protocol):raise ValueError('source mismatch')
    ledger=EpochExperimentRegistry(prior/'additional-budget.sqlite3',pin)
    try:records=ledger.records()
    finally:ledger.close()
    if [r['digest'] for r in records['attempts']]!=[r['job_digest'] for r in protocol['runs'][3:10]] or [r['state'] for r in records['attempts']]!=['INTERRUPTED']+['RESERVED']*6:
        raise ValueError('only six never-started roles may be newly admitted')
    interrupted=json.loads((prior/'colab/outcomes.json').read_text())['runs']
    if len(interrupted)!=1 or interrupted[0]['state']!='INTERRUPTED' or interrupted[0]['error']['type']!='TimeoutError':raise ValueError('different interruption requires separate review')
    item=interrupted[0];bundle=root/'artifacts'/item['run']
    envelope=json.loads((bundle/'signed_training_job.json').read_text());job=verify(envelope,pin)
    if sha(job)!=protocol['runs'][3]['job_digest'] or sha(job)!=item['job_digest']:raise ValueError('recovered job substitution')
    result=validate_bundle(bundle,job,pin)
    return root,pin,protocol,records,item,bundle,result


def prepare(key_path):
    root,pin,protocol,records,item,bundle,result=context();destination=root/'continuation-02';destination.mkdir(exist_ok=False)
    key=private_key(key_path)
    if key.public_key().public_bytes_raw().hex()!=pin:raise PermissionError('controller identity')
    recovery={'schema':'saal-public-native-recovery-v1','job_digest':item['job_digest'],'run':item['run'],
              'bundle_relative':bundle.relative_to(PROJECT).as_posix(),'result_digest':sha(result),
              'inventory_sha256':hashlib.sha256((bundle/'artifact_inventory.json').read_bytes()).hexdigest(),
              'original_outcome_digest':sha(item),'original_attempt_remains':'INTERRUPTED',
              'session_cleanup_verified':True,'retrained':False,'fresh_process_replay':True,
              'classification':'VALIDATED_MODEL_RECOVERED_AFTER_OPTIONAL_EXPORT_CLEANUP_TIMEOUT',
              'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    (destination/'signed_native_recovery.json').write_bytes(canonical(sign(recovery,key)))
    body={'schema':'saal-public-transport-continuation-v2','parent_protocol_digest':sha(protocol),'previous_ledger_digest':sha(records),
          'recovery_digest':sha(recovery),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'backend_sha256':hashlib.sha256((campaign.ROOT/'src/saal_rsi/epoch_backend_v4.py').read_bytes()).hexdigest(),
          'selected_job_digests':[r['job_digest'] for r in protocol['runs'][4:10]],'additional_training_reservations':6,
          'additional_reserved_seconds':6*3800,'deadline_seconds':24000,'full_job_digest':protocol['runs'][10]['job_digest'],
          'full_max_seconds':5000,'automatic_training_retries':0,'learning_parameters_changed':False,
          'reason':'Preserve validated seed83 adapter without retraining. Optional remote export cleanup may be deferred; final named-session stop and confirmation remain mandatory. Admit only six unstarted paired roles and the original unstarted full role.',
          'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    (destination/'signed_admission.json').write_bytes(canonical(sign(body,key)))


def run():
    root,pin,protocol,records,item,bundle,result=context();destination=root/'continuation-02'
    admission=verify(json.loads((destination/'signed_admission.json').read_text()),pin)
    recovery=verify(json.loads((destination/'signed_native_recovery.json').read_text()),pin)
    if admission['parent_protocol_digest']!=sha(protocol) or admission['previous_ledger_digest']!=sha(records) or admission['recovery_digest']!=sha(recovery) or recovery['result_digest']!=sha(result) or recovery['inventory_sha256']!=hashlib.sha256((bundle/'artifact_inventory.json').read_bytes()).hexdigest():raise ValueError('recovery evidence substituted')
    if admission['driver_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or admission['backend_sha256']!=hashlib.sha256((campaign.ROOT/'src/saal_rsi/epoch_backend_v4.py').read_bytes()).hexdigest():raise ValueError('transport source substituted')
    jobs=[json.loads((root/(r['name']+'-job.json')).read_text()) for r in protocol['runs'][4:10]]
    if [sha(verify(j,pin)) for j in jobs]!=admission['selected_job_digests'] or admission['learning_parameters_changed'] is not False or admission['automatic_training_retries']!=0:raise ValueError('changed learning or retry policy')
    execute_campaign(destination/'colab',jobs,pin,root/'artifacts',destination/'additional-budget.sqlite3',max_seconds=admission['deadline_seconds'])
    full=json.loads((root/(protocol['runs'][10]['name']+'-job.json')).read_text())
    if sha(verify(full,pin))!=admission['full_job_digest']:raise ValueError('full job substitution')
    free=shutil.disk_usage(PROJECT).free;required=protocol['resource_policy']['full_stage_minimum_local_free_bytes']
    if free<required:
        with (root/'full-storage-admission-rejected.json').open('xb') as f:f.write(canonical({'status':'ADMISSION_REJECTED','free_bytes':free,'required_bytes':required,'training_started':False,'qualification_status':'NOT_EVALUATED'}))
        raise RuntimeError('full recovery storage admission rejected')
    execute_campaign(root/'colab-full',[full],pin,root/'artifacts',root/'full-budget.sqlite3',max_seconds=admission['full_max_seconds'])


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run()
    else:raise ValueError('closed command')
