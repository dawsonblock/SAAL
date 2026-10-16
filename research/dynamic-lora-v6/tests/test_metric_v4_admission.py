import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_contract_v4 import new_job, POLICY, POLICY_DIGEST, worker_identity
from saal_rsi.adaptive_evaluator_v4 import validate_protocol, validate_jobs, validate_sources
from saal_rsi.adaptive_registry_v4 import AdaptiveRegistryV4
from saal_rsi.model_contract import sign
from saal_rsi.spec import sha


def _make_job(base):
    budget=copy.deepcopy(base['budget'])
    budget.update(campaign_id='v4_campaign',run_id='v4_run',max_runs=1,
                  total_lifecycle_seconds=5500,reserved_seconds=5000)
    return new_job(base['model_spec'], base['dataset'], base['adapter'],
        seed=base['seed'], budget=budget, steps=base['steps'],
        batch_size=base['batch_size'], accumulation=base['accumulation'],
        max_seconds=base['max_seconds'], campaign_stage='ENGINEERING',
        learning_rate=base['learning_rate'], head_learning_rate=base['head_learning_rate'],
        checkpoint_interval=base['checkpoint_interval'])


def _signed_pair(base, tmp_path):
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    job = _make_job(base)
    envelope = sign(job, key)
    name = 'engineering_'+str(job['seed'])
    budget = {'campaign_id': job['budget']['campaign_id'], 'max_runs': 1,
        'total_lifecycle_seconds': job['budget']['total_lifecycle_seconds'],
        'max_provisioning_attempts': 1, 'automatic_training_retries': 0,
        'financial_cap': False}
    sources = {'src/saal_rsi/adaptive_evaluator_v4.py': hashlib.sha256(
        __import__('pathlib').Path(__file__).parents[3].joinpath('src/saal_rsi/adaptive_evaluator_v4.py').read_bytes()).hexdigest()}
    protocol = {'schema':'saal-adaptive-campaign-v4','stage':'ENGINEERING','budget':budget,
        'job_digests':[sha(job)], 'runs':[{'name':name,'method':job['adapter']['method'],
            'seed':job['seed'],'job_digest':sha(job)}],
        'worker_identities':{'v4':worker_identity()},
        'numerical_metric_policy':copy.deepcopy(POLICY),
        'numerical_metric_policy_digest':POLICY_DIGEST,
        'colab_cli_version':'0.7.4','source_files':sources,
        'required_source_files':list(sources),
        'independent_generalization':'NOT_ESTABLISHED',
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    return key, pin, job, envelope, sign(protocol,key), protocol


def test_v4_protocol_binds_metric_worker_job_and_source(admitted_job, tmp_path):
    key,pin,job,envelope,signed,protocol = _signed_pair(admitted_job,tmp_path)
    verified = validate_jobs(protocol,[envelope],pin)
    assert verified == [job]
    validate_protocol(protocol)
    root=__import__('pathlib').Path(__file__).parents[3]
    assert validate_sources(protocol,root) is True
    downgraded=copy.deepcopy(protocol); downgraded['numerical_metric_policy_digest']='sha256:'+'0'*64
    with pytest.raises(ValueError): validate_protocol(downgraded)
    wrong=copy.deepcopy(job); wrong['numerical_metric_policy']=None
    with pytest.raises(ValueError): validate_jobs(protocol,[sign(wrong,key)],pin)


@pytest.mark.parametrize('mutation', [
    'retry', 'budget', 'duplicate', 'generalization', 'legacy',
    'unknown_field', 'malformed_required_sources', 'unsafe_source_name',
])
def test_v4_protocol_rejects_unsafe_or_downgraded_contract(admitted_job,tmp_path,mutation):
    _,_,_,_,_,protocol=_signed_pair(admitted_job,tmp_path)
    p=copy.deepcopy(protocol)
    if mutation=='retry': p['budget']['automatic_training_retries']=1
    elif mutation=='budget': p['budget']['max_provisioning_attempts']=2
    elif mutation=='duplicate': p['runs'].append(copy.deepcopy(p['runs'][0]));p['job_digests'].append(p['job_digests'][0])
    elif mutation=='generalization': p['independent_generalization']='ESTABLISHED'
    elif mutation=='legacy': p['schema']='saal-adaptive-campaign-v3'
    elif mutation=='unknown_field': p['authority_override'] = True
    elif mutation=='malformed_required_sources': p['required_source_files'] = [[]]
    elif mutation=='unsafe_source_name':
        p['source_files'] = {'../outside.py': '0' * 64}
        p['required_source_files'] = ['../outside.py']
    with pytest.raises(ValueError): validate_protocol(p)


def test_v4_registry_consumes_budget_once_and_never_resets(admitted_job,tmp_path):
    key,pin,job,envelope,protocol_envelope,protocol=_signed_pair(admitted_job,tmp_path)
    path=tmp_path/'v4-budget.sqlite3'
    ledger=AdaptiveRegistryV4(path,pin,protocol_envelope,[envelope],create=True)
    ledger.reserve_all([envelope]); ledger.begin_provisioning()
    with pytest.raises(ValueError): ledger.begin_provisioning()
    ledger.start(sha(job)); ledger.finish(sha(job),{'status':'interrupted'},False); ledger.end(False)
    export=ledger.records()
    assert export['campaign_state']=='FAILED'
    assert export['attempts'][0]['state']=='INTERRUPTED'
    assert export['provisioning_attempts']==1 and export['rollback_resistant'] is False
    assert export['qualification_status']=='NOT_EVALUATED' and export['external_deployment_allowed'] is False
    ledger.close()
    with pytest.raises(FileExistsError): AdaptiveRegistryV4(path,pin,protocol_envelope,[envelope],create=True)
    reopened=AdaptiveRegistryV4(path,pin,protocol_envelope,[envelope])
    with pytest.raises(ValueError): reopened.begin_provisioning()
    reopened.close()


def test_v4_registry_requires_all_reservations_before_provision(admitted_job,tmp_path):
    _,pin,_,envelope,protocol_envelope,_=_signed_pair(admitted_job,tmp_path)
    ledger=AdaptiveRegistryV4(tmp_path/'empty.sqlite3',pin,protocol_envelope,[envelope],create=True)
    with pytest.raises(ValueError): ledger.begin_provisioning()
    ledger.close()


def test_static_adalora_is_a_v4_only_no_reallocation_control(admitted_job):
    from saal_rsi.adaptive_contract_v4 import adapter_spec, validate_adapter
    adapter = adapter_spec('adalora-static-r12', total_steps=8, tinit=2, tfinal=2, delta=1)
    validate_adapter(adapter, 8)
    assert adapter['init_r'] == adapter['target_r'] == 12
    assert adapter['mask_policy'] == 'fixed-no-reallocation-v1'
    assert adapter['schedule']['total_step'] == 8
    with pytest.raises(ValueError):
        __import__('saal_rsi.adaptive_contract_v3', fromlist=['validate_job']).validate_job(
            {**admitted_job, 'adapter': adapter})


def test_v4_job_can_bind_static_adalora_without_reinterpreting_legacy(admitted_job):
    from saal_rsi.adaptive_contract_v4 import adapter_spec
    base = _make_job(admitted_job)
    adapter = adapter_spec('adalora-static-r12', total_steps=base['steps'],
                            tinit=1, tfinal=1, delta=1)
    adapter['target_modules'] = base['adapter']['target_modules']
    job = new_job(base['model_spec'], base['dataset'], adapter,
        seed=base['seed'], budget=base['budget'], steps=base['steps'],
        batch_size=base['batch_size'], accumulation=base['accumulation'],
        max_seconds=base['max_seconds'], campaign_stage=base['campaign_stage'],
        learning_rate=base['learning_rate'], head_learning_rate=base['head_learning_rate'],
        checkpoint_interval=base['checkpoint_interval'])
    assert job['schema'] == 'saal-adaptive-job-v4'
    assert job['adapter']['method'] == 'adalora-static-r12'
