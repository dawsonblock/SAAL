"""Reproduce retrospective learning selection, calibration and fresh public probes.

No training, protected evaluation, holdout access or qualification signature.
The new-mechanism protocol fixes prior candidates before obtaining predictions.
"""
from pathlib import Path
import argparse
import gc
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.spec import parse_unique_json,canonical,sha
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.research_selection import select_research_snapshots,POLICY
from saal_rsi.calibration_diagnostics import fit_temperature,selective_report
from saal_rsi.checkpoint_selection import metrics,paired_family_gain


def selections(root,controller_pin):
    rows=[];calibration=[]
    for bundle in sorted((root/'examples/pretrained_research').glob('artifacts-v*/*')):
        if not (bundle/'result.json').is_file():continue
        job=verify(parse_unique_json((bundle/'signed_training_job.json').read_bytes()),controller_pin)
        result=validate_bundle(bundle,job,controller_pin)
        retained={};mapping={}
        for name in ('initial','best','terminal'):
            raw=result['snapshots'][name];digest='sha256:'+hashlib.sha256((bundle/f'{name}.safetensors').read_bytes()).hexdigest()
            mapping[name]={'step':raw['step'],'checkpoint_digest':digest,'development':raw['metrics']['development']}
            if raw['step'] in retained and retained[raw['step']]['checkpoint_digest']!=digest:raise ValueError('same step has substituted weights')
            retained[raw['step']]=mapping[name]
        selected=select_research_snapshots(list(retained.values()))
        curve_best=max(result['selection_history'],key=lambda row:(row['metrics']['development']['family_macro_accuracy'],row['metrics']['development']['worst_family_accuracy'],row['metrics']['development']['accuracy'],-row['step']))
        rows.append({'bundle':bundle.relative_to(root).as_posix(),'job_digest':sha(job),'repository':job['model_spec']['model_repository'],'method':job['model_spec']['adapter_type'],'seed':job['seed'],
                     'historical_safety_best':mapping['best'],'retained_snapshot_learning_analysis':selected,
                     'highest_observed_curve_step':curve_best['step'],'highest_observed_curve_metrics':curve_best['metrics'],
                     'curve_best_weights_retained':curve_best['step'] in retained,
                     'policy_timing':'POST_HOC_PUBLIC_ANALYSIS_NOT_A_CHANGE_TO_SIGNED_JOB'})
        if job['model_spec']['model_kind']=='modernbert-seqcls-v1' and job['seed']==17 and job['steps']==500 and job['model_spec']['adapter_type']in ('head','lora'):
            dev=[r for r in job['dataset']['records'] if r['split']=='development'];reg=[r for r in job['dataset']['records'] if r['split']=='regression']
            outputs=result['snapshots']['terminal']['predictions']
            fit=fit_temperature(dev,outputs['development'],model_digest=mapping['terminal']['checkpoint_digest'])
            calibration.append({'bundle':bundle.relative_to(root).as_posix(),'fit':fit,
                                'regression':selective_report(reg,outputs['regression'],temperature=fit['temperature'])})
    return {'schema':'saal-corrective-public-analysis-v1','policy':POLICY,'runs':rows,'calibration':calibration,
            'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
            'limits':'Retrospective public evidence; highest learning-curve weights may not have been retained. No reconstruction or qualification of missing checkpoints. Temperature fitted on original public development and applied to public regression only.'}


def new_mechanism_probe(root,cache,pin,old_pin):
    import torch
    from saal_rsi.learning_backend import create_backend,verify_dependencies
    from saal_rsi.pretrained_registry import validate_snapshot
    torch.set_num_threads(2);verify_dependencies()
    e=root/'examples/corrective_research';protocol=verify(parse_unique_json((e/'signed-diagnostic-protocol.json').read_bytes()),pin)
    dataset=parse_unique_json((e/'new-public-diagnostics.json').read_bytes())
    if sha(dataset)!=protocol['dataset_digest'] or protocol['qualification_status']!='NOT_EVALUATED' or protocol['new_training']is not False:raise ValueError('public diagnostic protocol substitution')
    records=[r for r in dataset['records'] if r['split']=='development'];rows=[]
    for method,path in protocol['fixed_prior_candidates'].items():
        bundle=root/path
        job=verify(parse_unique_json((bundle/'signed_training_job.json').read_bytes()),old_pin);validate_bundle(bundle,job,old_pin)
        if job['seed']!=17 or job['model_spec']['adapter_type']!=method:raise ValueError('fixed prior candidate identity')
        spec=job['model_spec'];snapshot=Path(cache)/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
        backend=create_backend(spec,snapshot,job['seed']).load_pretrained();values={}
        for name in ('initial','terminal'):
            backend.restore_checkpoint(bundle/f'{name}.safetensors');outputs=backend.evaluate(records)
            values[name]={'checkpoint_sha256':hashlib.sha256((bundle/f'{name}.safetensors').read_bytes()).hexdigest(),'predictions':outputs,'metrics':metrics(records,outputs,spec['model_kind'])}
        backend.verify_frozen_weights();rows.append({'method':method,'job_digest':sha(job),'snapshots':values,'family_gain':paired_family_gain(values['initial']['metrics'],values['terminal']['metrics'])});del backend;gc.collect()
    # Keyword rule has no access to expected labels. Both controls are disclosed.
    def lexical(prompt):
        words=prompt.casefold();hits=[(words.find(word),label) for word,label in [('cheap','CHEAP'),('expense','CHEAP'),('inexpensive','CHEAP'),('cost','CHEAP'),('fast','FAST'),('quicker','FAST'),('delay','FAST'),('revoked','REFUSE'),('unspecified','ASK')]]
        hits=[x for x in hits if x[0]>=0];return min(hits)[1] if hits else 'CHEAP'
    def decision(label):return {'decision':label,'probabilities':[float(label==d) for d in ('CHEAP','FAST','ASK','REFUSE')]}
    rule=[decision(lexical(r['prompt'])) for r in records]
    from collections import Counter
    train=[r for r in dataset['records'] if r['split']=='train'];majority=Counter(r['expected_decision'] for r in train).most_common(1)[0][0]
    return {'schema':'saal-new-public-mechanism-results-v1','protocol_digest':sha(protocol),'record_ids':[r['record_id'] for r in records],
            'runs':rows,'lexical_control':{'predictions':rule,'metrics':metrics(records,rule,'modernbert-seqcls-v1')},
            'majority_control':{'label_fitted_on_public_train':majority,'metrics':metrics(records,[decision(majority) for _ in records],'modernbert-seqcls-v1')},
            'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED','external_deployment_allowed':False,
            'limits':'Six new operator-authored mechanisms, 72 related public development examples, fixed prior seed-17 checkpoints, no tuning or further training on this probe. One operator and synthetic construction do not establish independent generalization.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selection','new-probe'));p.add_argument('--old-pin',required=True);p.add_argument('--pin');p.add_argument('--cache');p.add_argument('--out',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    result=selections(root,a.old_pin) if a.mode=='selection' else new_mechanism_probe(root,a.cache,a.pin,a.old_pin)
    with Path(a.out).open('xb') as f:f.write(canonical(result))
    print('Public analysis completed; no qualification or activation issued')


if __name__=='__main__':main()
