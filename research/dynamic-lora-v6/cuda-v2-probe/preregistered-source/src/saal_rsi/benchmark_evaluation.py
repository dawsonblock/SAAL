"""Reusable public diagnostics, including explicit shortcut baselines; never qualification."""
from collections import Counter
import hashlib
from .dataset_registry import validate_manifest
from .model_contract import verify
from .model_backend import validate_bundle,bundle_digest
from .model_evaluation import inference_frozen
from .family_statistics import compare_families
from .benchmarks import lexical_shortcut
from .spec import sha


def evaluate_public(envelope,pin,bundle,dataset,split='regression'):
    if split not in ('development','regression') or dataset['origin']=='custodian-sequestered':raise ValueError('public diagnostics cannot access final holdout')
    validate_manifest(dataset);payload=verify(envelope,pin);validate_bundle(bundle,payload)
    if any(r['split']=='holdout' for r in dataset['records']):raise ValueError('holdout cannot be public selection data')
    records=[r for r in dataset['records'] if r['split']==split]
    if not records:raise ValueError('missing public benchmark split')
    result=inference_frozen(bundle,records,payload)
    # Majority label is derived ONLY from disclosed training labels.
    training=payload['dataset']['records'] if payload.get('schema')=='saal-model-job-v4' else payload['training']
    majority=Counter(r.get('expected_decision',r.get('label')) for r in training if r['split']=='train').most_common(1)[0][0]
    lexical=[lexical_shortcut(r['prompt']) for r in records]
    neural=compare_families(records,result['baseline'],result['candidate'],.05)
    controls={'majority':compare_families(records,result['baseline'],[majority]*len(records),.05),
              'lexical_shortcut':compare_families(records,result['baseline'],lexical,.05)}
    # Byte-multiset contradictions are a representational control for historical pooling.
    bags={};contradictions=[]
    for r,c in zip(records,result['candidate']):
        key=tuple(sorted(Counter(r['prompt'].encode()).items()))
        if key in bags and bags[key]['expected']!=r['expected_decision']:
            old=bags[key];contradictions.append({'left':old['id'],'right':r['record_id'],'left_expected':old['expected'],'right_expected':r['expected_decision'],
                                                'left_prediction':old['predicted'],'right_prediction':c,'both_correct':old['predicted']==old['expected'] and c==r['expected_decision']})
        else:bags[key]={'id':r['record_id'],'expected':r['expected_decision'],'predicted':c}
    validate_bundle(bundle,payload)
    return {'schema':'saal-public-benchmark-report-v2','dataset_digest':sha(dataset),'job_digest':sha(payload),'artifact_digest':bundle_digest(bundle,payload),'split':split,
            'records':records,'predictions':result,'neural':neural,'controls':controls,'byte_multiset_counterfactuals':contradictions,
            'lexical_shortcut_matches_or_exceeds_neural':controls['lexical_shortcut']['candidate_accuracy']>=neural['candidate_accuracy'],
            'qualification_status':'DIAGNOSTIC_ONLY','external_deployment_allowed':False,'model_activation_allowed':False,
            'limits':'Public, correlated synthetic controls. No independent population claim; not a protected holdout. Historical model parameters are unchanged.'}
