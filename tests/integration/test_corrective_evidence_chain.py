import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.evidence_chain import verify_chain, STAGES, ROLES, IDENTITY, GATES
from saal_rsi.attempt_authority import BINDINGS
from saal_rsi.model_contract import sign
from saal_rsi.spec import sha
from saal_rsi.promotion import authorize_external_activation, authorize_research_activation, ActivationDenied


def make():
    keys = {role: Ed25519PrivateKey.generate() for role in ('controller', 'custodian', 'evaluator', 'promotion', 'executor')}
    pins = {role: key.public_key().public_bytes_raw().hex() for role, key in keys.items()}
    identity = {k: sha(k) for k in IDENTITY}
    consumption = sign({'schema': 'saal-attempt-event-v1', 'kind': 'CONSUMED', 'bindings': {k: identity[k] for k in BINDINGS},
                        'seq': 2, 'attempt': 1, 'previous': sha('reservation'), 'enrollment_digest': sha('enrollment'),
                        'alpha': .025, 'time': 90}, keys['custodian'])
    chain = []
    details = [
        {'training_job_digest': sha('job'), 'parent_checkpoint_digest': sha('base'), 'status': 'NOT_EVALUATED'},
        {'artifact_inventory_digest': sha('inventory'), 'runtime_identity_digest': sha('runtime'), 'replay_digest': sha('replay'), 'attestation_scope': 'CONTROLLER_VERIFIED_ARTIFACTS_NOT_HOST_ATTESTATION'},
        {'consumption_digest': sha(consumption), 'metrics_digest': sha('metrics'), 'protocol_version': 'independent-family-v1'},
        {'evaluation_digest': None, 'gates': {k: True for k in GATES}, 'research_qualified': True, 'scope': 'BOUNDED_RESEARCH_ONLY'},
        {'qualification_digest': None, 'authorized': True, 'scope': 'SIMULATION_ONLY', 'expires_at': 1000},
        {'promotion_digest': None, 'enabled': True, 'scope': 'SIMULATION_ONLY', 'runtime_config_digest': sha('config')},
    ]
    for i, (stage, role, d) in enumerate(zip(STAGES, ROLES, details)):
        if i == 3: d['evaluation_digest'] = sha(chain[2])
        if i == 4: d['qualification_digest'] = sha(chain[3])
        if i == 5: d['promotion_digest'] = sha(chain[4])
        chain.append(sign({'schema': 'saal-artifact-authority-chain-v1', 'stage': stage, 'predecessor': sha(chain[-1]) if chain else None, 'identity': copy.deepcopy(identity), 'details': d}, keys[role]))
    return keys, pins, identity, consumption, chain


def resign_tail(chain, keys, start):
    for i in range(start, len(chain)):
        body = chain[i]['payload']
        body['predecessor'] = sha(chain[i-1]) if i else None
        if i >= 3: body['details'][('evaluation_digest','qualification_digest','promotion_digest')[i-3]] = sha(chain[i-1])
        chain[i] = sign(body, keys[ROLES[i]])


def test_chain_validity_does_not_issue_runtime_authority(tmp_path):
    keys, pins, identity, receipt, chain = make()
    result = verify_chain(chain, pins, identity, now=100, consumption=receipt)
    assert result['runtime_capability_issued'] is False and result['external_deployment_allowed'] is False
    with pytest.raises(ActivationDenied): authorize_external_activation(chain)
    with pytest.raises(ActivationDenied): authorize_research_activation(chain[0], chain[3], chain[4], evaluator_public_key=pins['evaluator'], anchor_public_key=pins['custodian'], human_public_key=pins['promotion'], replay_db=tmp_path/'approvals')


@pytest.mark.parametrize('index', range(6))
def test_signed_predecessor_substitution(index):
    keys, pins, identity, receipt, chain = make()
    chain[index]['payload']['identity']['candidate_digest'] = sha('other')
    chain[index] = sign(chain[index]['payload'], keys[ROLES[index]])
    with pytest.raises(PermissionError): verify_chain(chain, pins, identity, now=100, consumption=receipt)


@pytest.mark.parametrize('mutation', ['missing', 'reorder', 'impersonation', 'stale', 'remote_scope', 'bad_gate', 'same_roles', 'receipt', 'unknown_version'])
def test_no_authority_bypass(mutation):
    keys, pins, identity, receipt, chain = make()
    if mutation == 'missing': chain.pop(1)
    elif mutation == 'reorder': chain[1],chain[2] = chain[2],chain[1]
    elif mutation == 'impersonation': chain[2] = sign(chain[2]['payload'], keys['controller'])
    elif mutation == 'stale': chain[4]['payload']['details']['expires_at']=99; resign_tail(chain,keys,4)
    elif mutation == 'remote_scope': chain[4]['payload']['details']['scope']='EXTERNAL'; resign_tail(chain,keys,4)
    elif mutation == 'bad_gate': chain[3]['payload']['details']['gates']['retention']=False; resign_tail(chain,keys,3)
    elif mutation == 'same_roles': pins['promotion']=pins['controller']
    elif mutation == 'receipt': receipt=sign(receipt['payload']|{'bindings':{k:sha('other') for k in BINDINGS}},keys['custodian'])
    elif mutation == 'unknown_version': chain[2]['payload']['details']['protocol_version']='anything'; resign_tail(chain,keys,2)
    with pytest.raises((ValueError, PermissionError)): verify_chain(chain,pins,identity,now=100,consumption=receipt)
