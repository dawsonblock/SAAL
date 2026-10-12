"""Bounded public correction study. Every new budget is explicitly signed."""
from pathlib import Path
import sys, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import sha, canonical
from saal_rsi.model_contract import sign, verify
from saal_rsi.attempt_service import private_key
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.epoch_contract import new_job, SOURCES, worker_identity
from saal_rsi.epoch_backend_v2 import execute_campaign
from saal_rsi.corrective_corpus import validate

PROJECT=Path(__file__).parent
SEEDS=(47,83,109,157,211)
RUNS=[(f'{method}_{seed}',method,seed) for seed in SEEDS for method in ('head','lora')]+[('full_47','full',47)]
ADDITIONS=('behavior_screen.py','corrective_corpus.py','pretrained_qualification.py',
           'pretrained_inference.py','pretrained_qualification_cli.py')


def source_map():
    names=['src/saal_rsi/'+n for n in SOURCES+ADDITIONS]
    names+=['src/saal_rsi/epoch_backend_v2.py','src/saal_rsi/epoch_registry.py',
            'research/behavior-closure-v4/campaign.py',
            'tests/integration/test_pretrained_qualification_v6.py',
            'tests/integration/test_corrective_corpus_v4.py',
            'tests/integration/test_pretrained_inference_envelope.py',
            'docs/PRETRAINED_QUALIFICATION_V6.md',
            'configs/dependency-locks/pretrained-linux-x86_64-py313.txt']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def prepare(key_file):
    out=PROJECT/'experiment';out.mkdir(exist_ok=False)
    key=private_key(key_file);pin=key.public_key().public_bytes_raw().hex()
    data=json.loads((PROJECT/'dataset.json').read_text());audit=json.loads((PROJECT/'state-audit.json').read_text())
    parent=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text())
    validate(data,audit,parent);jobs=[]
    for name,method,seed in RUNS:
        spec=model_spec('answerdotai/ModernBERT-base',method=method,rank=4,backend='colab',
                        precision='float32',budget='behavior4_'+name,max_tokens=256)
        job=new_job(spec,data,sampler='complete-balanced-epochs-v1',seed=seed,steps=1000,
            batch_size=8,accumulation=2,learning_rate=.0003,head_learning_rate=.0003,
            encoder_learning_rate=.00001 if method=='full' else .0003,
            head_warmup_steps=100 if method=='full' else 0,max_seconds=2400,
            class_balance=True,checkpoint_interval=100)
        envelope=sign(job,key);jobs.append({'name':name,'method':method,'seed':seed,'job_digest':sha(job)})
        (out/(name+'-job.json')).write_bytes(canonical(envelope))
    protocol={'schema':'saal-public-corrective-learning-protocol-v1','source_files':source_map(),
        'worker_identity':worker_identity(),'dataset_digest':sha(data),'state_audit_digest':sha(audit),
        'runs':jobs,'selection':'Existing safety-first public-development selector; retain initial/best/terminal. Never select by regression or protected results.',
        'hypothesis':'Failure-informed short and composed public instruction supervision may improve feasibility, correction and provenance behavior. This is exploration after observing v3 errors.',
        'matched_comparisons':'Five paired head-only/fixed-LoRA seeds share initialization, data, complete epochs, head LR and 16000 sample exposures. LoRA adds adapters/dropout. One full-tuning seed is exploratory, with warmup and encoder LR specified separately.',
        'confounds_against_v3':['new corpus','twice the sample exposure per optimizer step'],
        'screen':{'complete_pairs':.9,'worst_family_accuracy':.8,'critical_proposals':0,'invalid_outputs':0,'four_class_coverage':True},
        'retention':'Report both exact original and newly added exposed populations. Do not merge old/new family scores to hide regressions.',
        'resource_policy':{'max_seconds_per_job':2400,'accumulation':2,'automatic_retries':0,
            'stage_one_jobs':10,'stage_one_deadline':39200,'full_stage_jobs':1,'full_stage_deadline':5000,
            'full_stage_minimum_local_free_bytes':5*1024**3,'financial_cap':False},
        'human_reviews':0,'independent_population':False,'protected_alpha_consumed':False,
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    (out/'signed_protocol.json').write_bytes(canonical(sign(protocol,key)))
    (out/'controller.pub').write_text(pin+'\n')
    print(json.dumps({'jobs':len(jobs),'dataset_digest':sha(data),'source_digest':sha(source_map())}))


def run(stage):
    out=PROJECT/'experiment';pin=(out/'controller.pub').read_text().strip()
    protocol=verify(json.loads((out/'signed_protocol.json').read_text()),pin)
    if protocol['source_files']!=source_map() or protocol['worker_identity']!=worker_identity():
        raise ValueError('preregistered source changed')
    if stage not in ('paired','full'):raise ValueError('closed stage')
    selected=protocol['runs'][:10] if stage=='paired' else protocol['runs'][10:]
    jobs=[json.loads((out/(r['name']+'-job.json')).read_text()) for r in selected]
    if [sha(verify(e,pin)) for e in jobs]!=[r['job_digest'] for r in selected]:
        raise ValueError('signed job substitution')
    if stage=='full' and shutil.disk_usage(PROJECT).free<protocol['resource_policy']['full_stage_minimum_local_free_bytes']:
        with (out/'full-storage-admission-rejected.json').open('xb') as f:
            f.write(canonical({'status':'ADMISSION_REJECTED','reason':'Insufficient local artifact recovery capacity',
                'free_bytes':shutil.disk_usage(PROJECT).free,'required_bytes':protocol['resource_policy']['full_stage_minimum_local_free_bytes'],
                'training_started':False,'qualification_status':'NOT_EVALUATED'}))
        raise RuntimeError('full tuning artifact recovery capacity is insufficient')
    return execute_campaign(out/('colab-'+stage),jobs,pin,out/'artifacts',out/(stage+'-budget.sqlite3'),
                            max_seconds=39200 if stage=='paired' else 5000)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise ValueError('closed command')
