"""Recompute exposed scores from authenticated native tensors and predictions."""
from pathlib import Path
import sys,json,statistics
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/reliable-learning-v3/tools'))
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import verify
from saal_rsi.epoch_artifacts import validate_bundle
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.behavior_screen import screen
from study_layout import load_runs,bundle_path


def macro_f1(measured):
    matrix=measured['confusion'];scores={}
    for label in ('CHEAP','FAST','ASK','REFUSE'):
        tp=matrix.get(label+'->'+label,0)
        fp=sum(v for key,v in matrix.items() if key.endswith('->'+label) and not key.startswith(label+'->'))
        fn=sum(v for key,v in matrix.items() if key.startswith(label+'->') and key!=label+'->'+label)
        scores[label]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return {'per_class_f1':scores,'macro_f1':sum(scores.values())/4}


def measure(bundle,pin,group):
    envelope=json.loads((bundle/'signed_training_job.json').read_text());job=verify(envelope,pin)
    result=validate_bundle(bundle,job,pin)
    original=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text())
    parent={r['record_id']:r for r in original['records']};snapshots={}
    for name,snapshot in result['snapshots'].items():
        splits={}
        for split in ('development','regression'):
            rows=[r for r in job['dataset']['records'] if r['split']==split];pred=snapshot['predictions'][split]
            for r in rows:
                if r['record_id'] in parent and r!=parent[r['record_id']]:raise ValueError('old exposed record changed')
            populations={'all':list(range(len(rows))),
                         'original':[i for i,r in enumerate(rows) if r['record_id'] in parent],
                         'added':[i for i,r in enumerate(rows) if r['record_id'] not in parent]}
            splits[split]={}
            for population,indices in populations.items():
                if not indices:continue
                m=metrics([rows[i] for i in indices],[pred[i] for i in indices],'modernbert-seqcls-v1')
                splits[split][population]={'records':len(indices),'metrics':m,'screen':screen(m),**macro_f1(m)}
        snapshots[name]={'step':snapshot['step'],'checkpoint':bundle.name+'/'+name+'.safetensors','splits':splits}
    return {'group':group,'run':bundle.name,'job_digest':sha(job),
            'method':job['model_spec']['adapter_type'],'seed':job['seed'],'sampler':job['training_policy']['sampler'],
            'snapshots':snapshots,'resource_accounting':result['resource_accounting'],
            'trainable_parameters':sum(v['numel'] for v in result['parameter_inventory'].values()),
            'raw_frozen_integrity':result['frozen_integrity'],'qualification_status':'NOT_EVALUATED'}


def report():
    prior=ROOT/'research/reliable-learning-v3';pin=(prior/'experiment/controller.pub').read_text().strip()
    old,protocol=load_runs(prior,pin);runs=[]
    for run in old:runs.append(measure(bundle_path(prior,run),pin,'v3-primary'))
    for group,folder in [('v3-head-control',PROJECT/'legacy-head/colab'),
                          ('v4-paired',PROJECT/'experiment/colab-paired'),
                          ('v4-paired-continuation',PROJECT/'experiment/continuation-01/colab'),
                          ('v4-paired-continuation-02',PROJECT/'experiment/continuation-02/colab'),
                          ('v4-full',PROJECT/'experiment/colab-full')]:
        path=folder/'outcomes.json'
        if not path.exists():continue
        base=PROJECT/'legacy-head/artifacts' if group=='v3-head-control' else PROJECT/'experiment/artifacts'
        for run in json.loads(path.read_text())['runs']:
            if run['state']=='COMPLETED':runs.append(measure(base/run['run'],pin,group))
    receipt=PROJECT/'experiment/continuation-02/signed_native_recovery.json'
    if receipt.exists():
        recovered=verify(json.loads(receipt.read_text()),pin)
        bundle=PROJECT/recovered['bundle_relative']
        measured=measure(bundle,pin,'v4-paired-recovered')
        if measured['job_digest']!=recovered['job_digest'] or recovered['original_attempt_remains']!='INTERRUPTED':raise ValueError('recovery identity')
        measured['transport_classification']=recovered['classification'];runs.append(measured)
    summaries={}
    for group in sorted({r['group'] for r in runs}):
        variants=sorted({(r['method'],r['sampler']) for r in runs if r['group']==group})
        for method,sampler in variants:
            population=[r for r in runs if r['group']==group and r['method']==method and r['sampler']==sampler]
            if not population:continue
            values=[r['snapshots']['terminal']['splits']['development']['all']['metrics']['accuracy'] for r in population]
            summaries[group+'/'+method+'/'+sampler]={'runs':len(values),'seeds':[r['seed'] for r in population],
                'mean_development_accuracy':statistics.mean(values),'standard_deviation':statistics.stdev(values) if len(values)>1 else None,
                'inferential_population_claim':False}
    for method in ('head','lora'):
        paired=[r for r in runs if r['group'] in ('v4-paired','v4-paired-continuation','v4-paired-continuation-02','v4-paired-recovered') and r['method']==method]
        if paired:
            if len({r['seed'] for r in paired})!=len(paired):raise ValueError('duplicate selected seed after continuation')
            values=[r['snapshots']['terminal']['splits']['development']['all']['metrics']['accuracy'] for r in paired]
            summaries['v4-complete-paired/'+method]={'runs':len(values),'seeds':[r['seed'] for r in paired],
                'mean_development_accuracy':statistics.mean(values),'standard_deviation':statistics.stdev(values) if len(values)>1 else None,
                'inferential_population_claim':False}
    return {'schema':'saal-behavior-closure-results-v1','runs':runs,'descriptive_summaries':summaries,
            'human_reviews':0,'independent_generalization':'NOT_ESTABLISHED','qualification_status':'NOT_EVALUATED',
            'external_deployment_allowed':False,
            'limits':'Only completed, physically recovered, authenticated runs. Old/new diagnostics reported separately. Public families are correlated; averages and F1 do not establish independent qualification.'}


if __name__=='__main__':
    target=Path(sys.argv[1])
    with target.open('xb') as f:f.write(canonical(report()))
