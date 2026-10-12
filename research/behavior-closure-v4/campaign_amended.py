"""Explicit pre-execution amendment; original protocol and jobs stay immutable."""
from pathlib import Path
import hashlib, json, sys, shutil
import campaign
ROOT=campaign.ROOT;PROJECT=campaign.PROJECT
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_contract import worker_identity
from saal_rsi.epoch_backend_v2 import execute_campaign

ALLOWED={'src/saal_rsi/pretrained_qualification.py','src/saal_rsi/pretrained_qualification_cli.py',
         'tests/integration/test_pretrained_qualification_v6.py','docs/PRETRAINED_QUALIFICATION_V6.md'}


def context():
    root=PROJECT/'experiment';pin=(root/'controller.pub').read_text().strip()
    protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin)
    current=campaign.source_map()
    changed={name:{'before':old,'after':current[name]} for name,old in protocol['source_files'].items() if old!=current[name]}
    if not set(changed)<=ALLOWED or set(current)!=set(protocol['source_files']) or protocol['worker_identity']!=worker_identity():
        raise ValueError('amendment cannot alter learning/data/worker sources')
    return root,pin,protocol,current,changed


def prepare(key_file):
    root,pin,protocol,current,changed=context()
    if any((root/name).exists() for name in ('paired-budget.sqlite3','full-budget.sqlite3','colab-paired','colab-full')):
        raise ValueError('only before any v4 execution/budget reservation')
    key=private_key(key_file)
    if key.public_key().public_bytes_raw().hex()!=pin:raise PermissionError('controller identity')
    body={'schema':'saal-public-source-amendment-v1','parent_protocol_digest':sha(protocol),
        'original_source_map_digest':sha(protocol['source_files']),'current_source_map':current,
        'changed_files':changed,'worker_identity':worker_identity(),'job_digests':[r['job_digest'] for r in protocol['runs']],
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reason':'Fix cross-platform qualification verification while retaining exact target enforcement during inference. No learning source, data, budget, seed, optimizer or checkpoint policy changes.',
        'learning_parameters_changed':False,'prior_v4_training_started':False,'protected_alpha_consumed':False,
        'fresh_qa_required':{'core':636,'corpus':38},'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    with (root/'signed_source_amendment.json').open('xb') as f:f.write(canonical(sign(body,key)))


def run(stage):
    root,pin,protocol,current,changed=context()
    a=verify(json.loads((root/'signed_source_amendment.json').read_text()),pin)
    if a['parent_protocol_digest']!=sha(protocol) or a['original_source_map_digest']!=sha(protocol['source_files']) or a['current_source_map']!=current or a['changed_files']!=changed or a['worker_identity']!=worker_identity() or a['learning_parameters_changed'] is not False:
        raise ValueError('signed source amendment substitution')
    # The initial registered driver is archived; a separate, signed full-stage
    # transport record authorizes this dispatcher revision before that job runs.
    if a['driver_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        transport=verify(json.loads((root/'signed_full_transport.json').read_text()),pin)
        if transport['parent_source_amendment_digest']!=sha(a) or transport['previous_driver_sha256']!=a['driver_sha256'] or transport['driver_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or transport['streaming_source_sha256']!=hashlib.sha256((ROOT/'src/saal_rsi/epoch_backend_v3.py').read_bytes()).hexdigest() or transport['full_job_digest']!=protocol['runs'][10]['job_digest'] or transport['learning_parameters_changed'] is not False:
            raise ValueError('full-stage transport identity substitution')
    if stage not in ('paired','full'):raise ValueError('closed stage')
    selected=protocol['runs'][:10] if stage=='paired' else protocol['runs'][10:]
    jobs=[json.loads((root/(r['name']+'-job.json')).read_text()) for r in selected]
    if [sha(verify(e,pin)) for e in jobs]!=[r['job_digest'] for r in selected] or a['job_digests']!=[r['job_digest'] for r in protocol['runs']]:
        raise ValueError('no job or learning changes permitted')
    if stage=='full' and shutil.disk_usage(PROJECT).free<protocol['resource_policy']['full_stage_minimum_local_free_bytes']:
        with (root/'full-storage-admission-rejected.json').open('xb') as f:
            f.write(canonical({'status':'ADMISSION_REJECTED','free_bytes':shutil.disk_usage(PROJECT).free,
                'required_bytes':protocol['resource_policy']['full_stage_minimum_local_free_bytes'],'training_started':False,
                'qualification_status':'NOT_EVALUATED'}))
        raise RuntimeError('full tuning recovery capacity unavailable')
    if stage=='full':
        from saal_rsi.epoch_backend_v3 import execute_campaign as streamed_campaign
        return streamed_campaign(root/'colab-full',jobs,pin,root/'artifacts',root/'full-budget.sqlite3',max_seconds=5000)
    return execute_campaign(root/'colab-paired',jobs,pin,root/'artifacts',root/'paired-budget.sqlite3',max_seconds=39200)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise ValueError('closed command')
