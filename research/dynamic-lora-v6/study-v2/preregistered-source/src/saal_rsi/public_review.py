"""Blind public review packets; signatures prove attribution, not human identity.

This never relaxes the existing reviewed-corpus export gate, fabricates reviews,
or converts public material into a protected/independent evaluation population.
"""
import base64
import hashlib
import json
from .learning_dataset import validate_dataset, diagnostic_audit
from .model_contract import verify, sign
from .spec import sha, canonical
from .validation import key_bytes


def packet(dataset):
    validate_dataset(dataset, 'modernbert-seqcls-v1')
    return {'schema': 'saal-public-blind-review-packet-v1', 'dataset_digest': sha(dataset),
            'records': [{'record_id': r['record_id'], 'prompt': r['prompt'], 'prompt_digest': sha(r['prompt']),
                         'decision': '', 'rationale': ''} for r in dataset['records']],
            'label_definitions': {'CHEAP': 'Select the named economy/CHEAP option when it is the current objective-optimal feasible option.',
                'FAST': 'Select the named express/FAST option when it is the current objective-optimal feasible option.',
                'ASK': 'Required information or the current objective is missing.',
                'REFUSE': 'Permission is absent/revoked/out of scope/expired, or no action is feasible.'},
            'reviewer_id': '', 'complete': False, 'human_reviews_claimed': 0,
            'limits': 'Proposed bounded semantics; review prompt meaning and ambiguity. Do not copy oracle labels. Public review is not independent qualification.'}


def review_body(dataset, draft, *, role='reviewer'):
    validate_dataset(dataset, 'modernbert-seqcls-v1')
    if role not in ('reviewer', 'adjudicator') or draft.get('schema') != 'saal-public-blind-review-packet-v1' or draft.get('dataset_digest') != sha(dataset):
        raise ValueError('stale review dataset/role')
    if draft.get('complete') is not True or type(draft.get('reviewer_id')) is not str or not 1 <= len(draft['reviewer_id']) <= 100:
        raise ValueError('incomplete review or reviewer identity')
    originals = {r['record_id']: r for r in dataset['records']}; answers = {}
    if type(draft.get('records')) is not list: raise ValueError('review records')
    for r in draft['records']:
        if type(r) is not dict or set(r) != {'record_id', 'prompt', 'prompt_digest', 'decision', 'rationale'}:
            raise ValueError('review record schema')
        rid = r['record_id']
        if rid not in originals or rid in answers or r['prompt'] != originals[rid]['prompt'] or r['prompt_digest'] != sha(r['prompt']):
            raise ValueError('duplicate/substituted review prompt')
        if r['decision'] not in ('CHEAP', 'FAST', 'ASK', 'REFUSE', 'REJECT') or type(r['rationale']) is not str or not 1 <= len(r['rationale'].strip().encode()) <= 2048:
            raise ValueError('review decision and rationale required')
        answers[rid] = {'decision': r['decision'], 'rationale': r['rationale'], 'prompt_digest': r['prompt_digest']}
    if set(answers) != set(originals): raise ValueError('full review coverage required')
    return {'schema': 'saal-public-review-batch-v2', 'dataset_digest': sha(dataset),
            'reviewer_id': draft['reviewer_id'], 'role': role, 'decisions': answers}


def sign_draft(dataset, draft, key, *, role='reviewer'):
    return sign(review_body(dataset, draft, role=role), key)


def assess(dataset, policy, reviews, semantic_review, adjudication=None):
    validate_dataset(dataset, 'modernbert-seqcls-v1')
    if type(policy) is not dict or set(policy) != {'schema', 'reviewers', 'adjudicator'} or policy['schema'] != 'saal-public-review-policy-v2' or type(policy['reviewers']) is not dict or len(policy['reviewers']) != 2:
        raise ValueError('two enrolled reviewers required')
    pins = [key_bytes(p).hex() for p in policy['reviewers'].values()] + [key_bytes(policy['adjudicator']).hex()]
    if len(set(pins)) != 3: raise ValueError('distinct reviewer/adjudicator keys required')
    if type(reviews) is not list or len(reviews) != 2: raise ValueError('review quorum missing')
    bodies = {}
    ids = {r['record_id'] for r in dataset['records']}
    prompt_digests = {r['record_id']: sha(r['prompt']) for r in dataset['records']}
    def checked(body, role):
        if type(body) is not dict or set(body) != {'schema', 'dataset_digest', 'reviewer_id', 'role', 'decisions'} or body['schema'] != 'saal-public-review-batch-v2' or body['role'] != role or body['dataset_digest'] != sha(dataset) or set(body['decisions']) != ids:
            raise ValueError('stale/incomplete review batch')
        for rid, answer in body['decisions'].items():
            if type(answer) is not dict or set(answer) != {'decision', 'rationale', 'prompt_digest'} or answer['decision'] not in ('CHEAP', 'FAST', 'ASK', 'REFUSE', 'REJECT') or answer['prompt_digest'] != prompt_digests[rid] or type(answer['rationale']) is not str or not 1 <= len(answer['rationale'].strip().encode()) <= 2048:
                raise ValueError('invalid signed review answer')
        return body
    for envelope in reviews:
        identity = envelope['payload']['reviewer_id']
        if identity not in policy['reviewers'] or identity in bodies: raise ValueError('review identity duplication/substitution')
        bodies[identity] = checked(verify(envelope, policy['reviewers'][identity]), 'reviewer')
    resolved = checked(verify(adjudication, policy['adjudicator']), 'adjudicator') if adjudication is not None else None
    conflicts = []
    for r in dataset['records']:
        rid = r['record_id']; a, b = [body['decisions'][rid]['decision'] for body in bodies.values()]
        if 'REJECT' in (a, b): raise ValueError('rejected record requires new corpus version')
        if a != b:
            conflicts.append(rid)
            if resolved is None: raise ValueError('unresolved disagreement')
            label = resolved['decisions'][rid]['decision']
        else: label = a
        if label != r['expected_decision']: raise ValueError('review/oracle disagreement requires a new dataset and review cycle')
    if semantic_review is None: raise ValueError('semantic split review missing')
    screen = diagnostic_audit(dataset)
    semantic = verify(semantic_review, policy['adjudicator'])
    if type(semantic) is not dict or set(semantic) != {'schema', 'dataset_digest', 'lexical_screen_digest', 'approved_public_use', 'rationale', 'independent_population'} or semantic['schema'] != 'saal-public-semantic-review-v2' or semantic['dataset_digest'] != sha(dataset) or semantic['lexical_screen_digest'] != sha(screen) or semantic['approved_public_use'] is not True or semantic['independent_population'] is not False or type(semantic['rationale']) is not str or not semantic['rationale'].strip():
        raise ValueError('stale, unapproved or inflated semantic review')
    return {'schema': 'saal-public-review-assessment-v2', 'dataset_digest': sha(dataset),
            'signed_quorum_records': len(ids), 'disagreements_adjudicated': conflicts,
            'review_evidence_digests': [sha(e) for e in reviews] + ([sha(adjudication)] if adjudication else []) + [sha(semantic_review)],
            'human_identity_verified': False, 'operational_independence_established': False,
            'existing_reviewed_corpus_export_gate_unchanged': True,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def html_packet(dataset):
    """Offline blind UI; no oracle answers, external requests or automatic choices."""
    draft = json.dumps(packet(dataset), ensure_ascii=True).replace('<', '\\u003c')
    script = "const d=" + draft + ";" + """
let i=0; const $=x=>document.getElementById(x);
function show(){const r=d.records[i]; $('progress').textContent=(i+1)+' / '+d.records.length;
 $('prompt').textContent=r.prompt; $('decision').value=r.decision; $('rationale').value=r.rationale;}
function save(){d.records[i].decision=$('decision').value; d.records[i].rationale=$('rationale').value;}
$('definitions').textContent=JSON.stringify(d.label_definitions,null,2);
$('prev').addEventListener('click',()=>{save();i=Math.max(0,i-1);show();});
$('next').addEventListener('click',()=>{save();i=Math.min(d.records.length-1,i+1);show();});
$('restore').addEventListener('change',async()=>{try{
 const x=JSON.parse(await $('restore').files[0].text());
 if(x.schema!==d.schema||x.dataset_digest!==d.dataset_digest||!Array.isArray(x.records)||x.records.length!==d.records.length)throw Error('Different or incomplete corpus');
 const labels=['','CHEAP','FAST','ASK','REFUSE','REJECT'];
 x.records.forEach((r,k)=>{const o=d.records[k];if(r.record_id!==o.record_id||r.prompt!==o.prompt||r.prompt_digest!==o.prompt_digest||!labels.includes(r.decision)||typeof r.rationale!=='string'||new TextEncoder().encode(r.rationale).length>2048)throw Error('Changed prompt or malformed review');});
 if(typeof x.reviewer_id!=='string'||x.reviewer_id.length>100)throw Error('Reviewer identity');
 x.records.forEach((r,k)=>{d.records[k].decision=r.decision;d.records[k].rationale=r.rationale;});
 $('reviewer').value=x.reviewer_id;i=Math.max(0,d.records.findIndex(r=>!r.decision||!r.rationale.trim()));show();$('status').textContent='Draft restored. No signature or qualification created.';
 }catch(e){$('status').textContent='Import rejected: '+e.message;}});
$('download').addEventListener('click',()=>{save();d.reviewer_id=$('reviewer').value;
 d.complete=d.records.every(r=>r.decision&&r.rationale.trim());
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));
 a.download='public-review-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});show();
"""
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    return ('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'sha256-' + script_hash + '\'; style-src \'unsafe-inline\'; connect-src \'none\'; base-uri \'none\'; form-action \'none\'">'
            '<title>SAAL public blind review</title><style>body{font:18px system-ui;max-width:900px;margin:40px auto;padding:16px}pre{white-space:pre-wrap}textarea{width:100%;height:120px}button,select,input{font:inherit;margin:8px;padding:8px}</style>'
            '<h1>Public blind review</h1><p>Choose a label from the actual instruction and explain why. No choice is preselected. Review signatures cannot prove independent human identity.</p>'
            '<details><summary>Proposed decision semantics</summary><pre id="definitions"></pre><p>Invalid authority or known impossibility requires refusal before checking missing information. Confidence alone is not missing task information.</p></details>'
            '<label>Reviewer ID <input id="reviewer"></label><label>Restore a saved draft <input id="restore" type="file" accept="application/json"></label><p id="status" role="status"></p><p id="progress"></p><pre id="prompt"></pre>'
            '<select id="decision"><option value="">Choose manually</option><option>CHEAP</option><option>FAST</option><option>ASK</option><option>REFUSE</option><option>REJECT</option></select>'
            '<textarea id="rationale" aria-label="Review rationale"></textarea><button id="prev">Previous</button><button id="next">Next</button><button id="download">Download review draft</button><script>' + script + '</script>')
