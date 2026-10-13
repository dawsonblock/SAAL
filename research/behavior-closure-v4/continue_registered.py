"""Explicit additional training admission after a lost controller; no budget reset."""
from pathlib import Path
import hashlib,json,sys,subprocess
import campaign_amended as campaign
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.epoch_backend_v2 import execute_campaign

PROJECT=Path(__file__).parent


def original():
    root,pin,protocol,current,changed=campaign.context()
    amendment=verify(json.loads((root/'signed_source_amendment.json').read_text()),pin)
    if amendment['current_source_map']!=current or amendment['parent_protocol_digest']!=sha(protocol):
        raise ValueError('source amendment mismatch')
    reg=EpochExperimentRegistry(root/'paired-budget.sqlite3',pin)
    try:records=reg.records()
    finally:reg.close()
    attempts=records['attempts']
    expected=protocol['runs'][:10]
    if [a['digest'] for a in attempts]!=[r['job_digest'] for r in expected]:raise ValueError('original attempt inventory')
    if [a['state'] for a in attempts]!=['COMPLETED']*3+['INTERRUPTED']+['RESERVED']*6:
        raise ValueError('continuation only for the recorded interruption; no successful run retry')
    outcomes=json.loads((root/'colab-paired/outcomes.json').read_text())
    if [r['job_digest'] for r in outcomes['runs']]!=[r['job_digest'] for r in expected[:4]] or [r['state'] for r in outcomes['runs']]!=['COMPLETED']*3+['INTERRUPTED']:
        raise ValueError('original recovered evidence inventory')
    return root,pin,protocol,records,outcomes


def prepare(key_path):
    root,pin,protocol,records,outcomes=original()
    destination=root/'continuation-01';destination.mkdir(exist_ok=False)
    key=private_key(key_path)
    if key.public_key().public_bytes_raw().hex()!=pin:raise PermissionError('controller identity')
    body={'schema':'saal-public-explicit-continuation-v1','parent_protocol_digest':sha(protocol),
          'old_registry_digest':sha(records),'old_outcomes_digest':sha(outcomes),
          'old_budget_sha256':hashlib.sha256((root/'paired-budget.sqlite3').read_bytes()).hexdigest(),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'selected_job_digests':[r['job_digest'] for r in protocol['runs'][3:10]],
          'additional_training_reservations':7,'additional_reserved_seconds':7*(2400+1400),
          'campaign_deadline_seconds':27800,'automatic_retries':0,
          'reason':'Controller output pipe interrupted. Three recovered runs preserved; seed83 LoRA interrupted and six never started. One fresh, explicit additional budget is admitted; original reservations and negative outcome remain intact.',
          'learning_parameters_changed':False,'protected_alpha_consumed':False,
          'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    (destination/'signed_admission.json').write_bytes(canonical(sign(body,key)))


def run():
    root,pin,protocol,records,outcomes=original();destination=root/'continuation-01'
    admission=verify(json.loads((destination/'signed_admission.json').read_text()),pin)
    if admission['parent_protocol_digest']!=sha(protocol) or admission['old_registry_digest']!=sha(records) or admission['old_outcomes_digest']!=sha(outcomes) or admission['old_budget_sha256']!=hashlib.sha256((root/'paired-budget.sqlite3').read_bytes()).hexdigest() or admission['driver_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('continuation evidence or driver substitution')
    jobs=[json.loads((root/(r['name']+'-job.json')).read_text()) for r in protocol['runs'][3:10]]
    if [sha(verify(j,pin)) for j in jobs]!=admission['selected_job_digests'] or admission['learning_parameters_changed'] is not False or admission['automatic_retries']!=0:
        raise ValueError('new job parameters or retries prohibited')
    execute_campaign(destination/'colab',jobs,pin,root/'artifacts',destination/'additional-budget.sqlite3',max_seconds=admission['campaign_deadline_seconds'])
    subprocess.run([sys.executable,str(PROJECT/'campaign_amended.py'),'run','full'],check=True,timeout=5100)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run()
    else:raise ValueError('closed command')
