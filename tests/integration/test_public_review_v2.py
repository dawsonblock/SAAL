"""Synthetic key/answer fixtures test plumbing; these are NOT human reviews."""
import copy
import hashlib
import base64
import re
import shutil
import subprocess
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.learning_dataset import semantic_pilot, diagnostic_audit
from saal_rsi.public_review import packet, html_packet, sign_draft, assess
from saal_rsi.model_contract import sign
from saal_rsi.spec import sha


@pytest.fixture(scope='module')
def dataset(): return semantic_pilot()


@pytest.fixture
def evidence(dataset):
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    policy = {'schema': 'saal-public-review-policy-v2', 'reviewers': {'toy_a': keys[0].public_key().public_bytes_raw().hex(), 'toy_b': keys[1].public_key().public_bytes_raw().hex()}, 'adjudicator': keys[2].public_key().public_bytes_raw().hex()}
    drafts = []
    for i in range(2):
        draft = packet(dataset); draft['reviewer_id'] = ['toy_a', 'toy_b'][i]; draft['complete'] = True
        for row, r in zip(draft['records'], dataset['records']):
            row['decision'] = r['expected_decision']; row['rationale'] = 'Synthetic unit-test answer, not a human review.'
        drafts.append(draft)
    semantic = {'schema': 'saal-public-semantic-review-v2', 'dataset_digest': sha(dataset), 'lexical_screen_digest': sha(diagnostic_audit(dataset)), 'approved_public_use': True, 'rationale': 'Synthetic test fixture only.', 'independent_population': False}
    return keys, policy, drafts, semantic


def test_blind_packet_has_no_answers_or_automatic_reviews(dataset):
    draft = packet(dataset)
    assert all(r['decision'] == r['rationale'] == '' for r in draft['records'])
    assert not draft['complete'] and draft['human_reviews_claimed'] == 0
    page = html_packet(dataset)
    assert 'expected_decision' not in page and 'oracle_payload' not in page
    assert "connect-src 'none'" in page


def test_html_prompt_injection_stays_text(dataset):
    data = copy.deepcopy(dataset); data['records'][0]['prompt'] = '</script><img src=x onerror=alert(1)>'
    page = html_packet(data)
    assert '</script><img' not in page and '\\u003c/script>' in page
    assert 'textContent=r.prompt' in page


def test_review_ui_script_export_resume_and_rejects_tamper(dataset):
    node = shutil.which('node')
    if node is None: pytest.skip('Node unavailable; live browser test is separate')
    page = html_packet(dataset); script = re.search(r'<script>(.*?)</script>', page, re.S).group(1)
    expected = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert 'sha256-' + expected in page
    harness = r'''
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const elements={},events={};let exported;
function el(id){return elements[id]??=( {value:'',textContent:'',files:[],addEventListener:(name,fn)=>{events[id+':'+name]=fn;}});}
class BlobMock{constructor(parts){exported=JSON.parse(parts.join(''));}}
const context={document:{getElementById:el,createElement:()=>({click(){}})},Blob:BlobMock,
 URL:{createObjectURL:()=> 'blob:unit-test',revokeObjectURL(){}},TextEncoder,setTimeout(){}};
vm.runInNewContext(fs.readFileSync(0,'utf8'),context);
assert.equal(el('decision').value,'');assert.match(el('progress').textContent,/^1 \/ /);
el('decision').value='REJECT';el('rationale').value='UI logic test, not a human review.';
events['next:click']();assert.match(el('progress').textContent,/^2 \/ /);
el('reviewer').value='UI_TEST_ONLY';events['download:click']();
assert.equal(exported.complete,false);assert.equal(exported.human_reviews_claimed,0);
const draft=JSON.parse(JSON.stringify(exported));draft.complete=true;
(async()=>{el('restore').files=[{text:async()=>JSON.stringify(draft)}];await events['restore:change']();
 assert.match(el('status').textContent,/Draft restored/);events['download:click']();assert.equal(exported.complete,false);
 draft.records[0].prompt='tampered';el('restore').files=[{text:async()=>JSON.stringify(draft)}];await events['restore:change']();
 assert.match(el('status').textContent,/Import rejected/);console.log('offline UI logic passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    # Unit-test DOM mock only; no alternate browser/computer automation.
    result = subprocess.run([node, '-e', harness], input=script, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_complete_signed_quorum_is_still_not_independent_or_qualified(dataset, evidence):
    keys, policy, drafts, semantic = evidence
    result = assess(dataset, policy, [sign_draft(dataset, d, k) for d, k in zip(drafts, keys)], sign(semantic, keys[2]))
    assert result['signed_quorum_records'] == len(dataset['records'])
    assert not result['human_identity_verified'] and not result['operational_independence_established']
    assert result['qualification_status'] == 'NOT_EVALUATED' and not result['external_deployment_allowed']


@pytest.mark.parametrize('mutation', ['incomplete', 'prompt', 'decision', 'rationale', 'duplicate', 'stale'])
def test_signing_refuses_invalid_or_unfinished_review(dataset, evidence, mutation):
    keys, _, drafts, _ = evidence; d = copy.deepcopy(drafts[0])
    if mutation == 'incomplete': d['complete'] = False
    elif mutation == 'prompt': d['records'][0]['prompt'] = 'replacement'
    elif mutation == 'decision': d['records'][0]['decision'] = 'ALLOW_ANYTHING'
    elif mutation == 'rationale': d['records'][0]['rationale'] = ' '
    elif mutation == 'duplicate': d['records'][-1] = d['records'][0]
    else: d['dataset_digest'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError): sign_draft(dataset, d, keys[0])


@pytest.mark.parametrize('mutation', ['same_key', 'quorum', 'signature', 'disagreement', 'rejection', 'semantic_missing', 'semantic_stale', 'independence'])
def test_forged_or_unresolved_reviews_cannot_be_admitted(dataset, evidence, mutation):
    keys, policy, drafts, semantic = evidence
    if mutation == 'same_key': policy['adjudicator'] = policy['reviewers']['toy_a']
    elif mutation == 'disagreement': drafts[1]['records'][0]['decision'] = 'ASK' if drafts[0]['records'][0]['decision'] != 'ASK' else 'REFUSE'
    elif mutation == 'rejection': drafts[0]['records'][0]['decision'] = 'REJECT'
    elif mutation == 'semantic_stale': semantic['lexical_screen_digest'] = 'sha256:' + '0' * 64
    elif mutation == 'independence': semantic['independent_population'] = True
    reviews = [sign_draft(dataset, d, k) for d, k in zip(drafts, keys)]
    if mutation == 'quorum': reviews.pop()
    elif mutation == 'signature': reviews[0]['signature_b64'] = reviews[1]['signature_b64']
    signed_semantic = None if mutation == 'semantic_missing' else sign(semantic, keys[2])
    with pytest.raises(ValueError): assess(dataset, policy, reviews, signed_semantic)
