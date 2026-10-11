"""Family protocol with signed custody review; local roles remain an operational assumption."""
from .dataset_registry import validate_manifest,leakage_audit,legacy_projection
from .family_statistics import V2_RULES,compare_families
from .model_contract import verify,sign
from .model_backend import bundle_digest
from .spec import sha,canonical
from .validation import key_bytes

REVIEW_FIELDS={'schema','dataset_digests','audit_digest','decision','sampling_justification','operational_separation','test_assumptions','flag_resolutions'}


def create_review(public,holdout,key,*,decision='diagnostic-only',sampling_justification='Public synthetic examples; population independence and separate custody are not established.',operational_separation='not-established',test_assumptions='No independent family sampling or sign-null assumptions established for public synthetic diagnostics.',flag_resolutions=None):
    audit=leakage_audit(public,holdout)
    body={'schema':'saal-custody-review-v2','dataset_digests':audit['dataset_digests'],'audit_digest':sha(audit),
          'decision':decision,'sampling_justification':sampling_justification,'operational_separation':operational_separation,'test_assumptions':test_assumptions,'flag_resolutions':{} if flag_resolutions is None else flag_resolutions}
    signed=sign(body,key);verify_review(signed,key.public_key().public_bytes_raw().hex(),public,holdout);return signed


def verify_review(envelope,pin,public,holdout):
    body=verify(envelope,pin);audit=leakage_audit(public,holdout)
    if type(body)is not dict or set(body)!=REVIEW_FIELDS or body['schema']!='saal-custody-review-v2':raise ValueError('custody review schema')
    if body['dataset_digests']!=audit['dataset_digests'] or body['audit_digest']!=sha(audit):raise ValueError('custody review dataset/audit substitution')
    if body['decision'] not in ('diagnostic-only','approved-independent') or body['operational_separation'] not in ('not-established','independently-managed'):raise ValueError('custody decision/separation')
    if type(body['sampling_justification'])is not str or not 20<=len(body['sampling_justification'])<=4000:raise ValueError('sampling justification required')
    if type(body['test_assumptions'])is not str or not 20<=len(body['test_assumptions'])<=4000:raise ValueError('preregistered sign-null justification required')
    resolutions=body['flag_resolutions'];flags={sha(f) for f in audit['review_flags']}
    if type(resolutions)is not dict or not set(resolutions)<=flags or any(type(v)is not str or not 20<=len(v)<=2000 for v in resolutions.values()):raise ValueError('similarity review resolutions')
    if body['decision']=='approved-independent':
        if holdout['origin']!='custodian-sequestered' or body['operational_separation']!='independently-managed' or set(resolutions)!=flags:raise ValueError('independent sampling/custody and complete similarity review required')
    return {'independence_verified':body['decision']=='approved-independent','leakage_review_complete':set(resolutions)==flags,'review':body,'audit':audit}


def family_plan(payload,bundle,public,holdout,review,custodian_pin,controller_pin):
    from .model_evaluation import learning_identity
    validate_manifest(public);validate_manifest(holdout)
    limit=payload['model_spec']['max_sequence_bytes'] if payload.get('schema')=='saal-model-job-v4' else 128
    if any(len(r['prompt'].encode())>limit for m in (public,holdout) for r in m['records']):raise ValueError('model prompt bound; truncation prohibited')
    if {r['split'] for r in holdout['records']}!={'holdout'} or any(r['split']=='holdout' for r in public['records']):raise ValueError('private/public partition inventory')
    if payload.get('schema')=='saal-model-job-v4':
        from .neural_artifacts import validate_neural_bundle
        validate_neural_bundle(bundle,payload,controller_pin=controller_pin,replay=True)
        if payload['dataset_digest']!=sha(public) or payload['dataset']!=public:raise ValueError('signed public dataset/model binding')
    else:
        if payload.get('schema')=='saal-model-job-v3' and payload['dataset_digest']!=sha(public):raise ValueError('signed public dataset digest mismatch')
        records={r['id']:r for r in legacy_projection(public,'train')+legacy_projection(public,'development')}
        for split in ('training','adapter_training','development'):
            if any(records.get(r['id'])!=r for r in payload[split]):raise ValueError('job/dataset lineage binding')
        if {r['id'] for r in payload['development']}!={r['record_id'] for r in public['records'] if r['split']=='development'}:raise ValueError('development inventory binding')
    if not any(r['split']=='regression' for r in public['records']):raise ValueError('regression suite required')
    verify_review(review,custodian_pin,public,holdout)
    if key_bytes(controller_pin)==key_bytes(custodian_pin):raise ValueError('distinct controller/custodian identities required')
    return {'schema':'saal-model-sealed-plan-v2','job_digest':sha(payload),'artifact_digest':bundle_digest(bundle,payload),
            'public_dataset':public,'holdout_dataset':holdout,'custody_review':review,'custodian_pin':key_bytes(custodian_pin).hex(),'controller_pin':key_bytes(controller_pin).hex(),
            'rules':V2_RULES.copy(),'identity':learning_identity(),'regression_policy':'no-family-accuracy-loss-v1'}


def validate_family_plan(plan,payload,bundle):
    if type(plan)is not dict or set(plan)!={'schema','job_digest','artifact_digest','public_dataset','holdout_dataset','custody_review','custodian_pin','controller_pin','rules','identity','regression_policy'}:raise ValueError('family plan schema')
    if plan!=family_plan(payload,bundle,plan['public_dataset'],plan['holdout_dataset'],plan['custody_review'],plan['custodian_pin'],plan['controller_pin']):raise ValueError('family plan/protocol identity')


def family_metrics(plan,baseline,candidate,regression_baseline,regression_candidate,alpha):
    review=verify_review(plan['custody_review'],plan['custodian_pin'],plan['public_dataset'],plan['holdout_dataset'])
    metrics=compare_families(plan['holdout_dataset']['records'],baseline,candidate,alpha,
        independence_verified=review['independence_verified'],leakage_review_complete=review['leakage_review_complete'],origin=plan['holdout_dataset']['origin'])
    regression=[r for r in plan['public_dataset']['records'] if r['split']=='regression']
    retention=compare_families(regression,regression_baseline,regression_candidate,alpha)
    retained=all(v['candidate_accuracy']>=v['baseline_accuracy'] for v in retention['per_family'].values())
    metrics['gates']['regression_zero_critical_errors']=not any(retention['behavior_errors'].get(k,0) for k in ('injection','proxy','refusal','clarification','correction')) and retention['unauthorized_proposals']['count']==0
    metrics['gates']['prior_capability_retention']=retained;metrics['research_qualified']=all(metrics['gates'].values())
    metrics['regression']={'policy':plan['regression_policy'],'retained':retained,'metrics':retention}
    return metrics
