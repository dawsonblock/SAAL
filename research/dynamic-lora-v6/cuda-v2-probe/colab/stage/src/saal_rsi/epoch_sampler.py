"""Deterministic complete epochs; evaluation records cannot enter the stream."""
import collections,random
from .pretrained_contract import DECISIONS
from .spec import sha

def balanced_counts(records):
    if not records or any(r['split']!='train' for r in records) or len({r['record_id'] for r in records})!=len(records):raise ValueError('training-only unique population')
    counts=collections.Counter(r['expected_decision'] for r in records)
    if set(counts)!=set(DECISIONS) or len(set(counts.values()))!=1:raise ValueError('complete balanced epochs require equal four-class counts')
    return counts

def sample_ids(records,mode,seed,count):
    if type(seed)is not int or type(count)is not int or not 1<=count<=512000 or any(r['split']!='train' for r in records):raise ValueError('sampler budget/partition')
    rng=random.Random(seed)
    if mode=='uniform-v1':return [rng.choice(records)['record_id'] for _ in range(count)]
    if mode!='complete-balanced-epochs-v1':raise ValueError('sampler mode')
    counts=balanced_counts(records);buckets={d:[r['record_id'] for r in records if r['expected_decision']==d] for d in DECISIONS};out=[]
    while len(out)<count:
        for ids in buckets.values():rng.shuffle(ids)
        for i in range(counts[DECISIONS[0]]):
            for d in DECISIONS:out.append(buckets[d][i])
    return out[:count]

def diagnostic_ids(records,seed,per_class):
    if any(r['split']!='train' for r in records):raise ValueError('diagnostics must be training-only')
    rng=random.Random(seed);ids=[]
    for d in DECISIONS:
        population=[r['record_id'] for r in records if r['expected_decision']==d]
        if not population:raise ValueError('training diagnostic class absent')
        ids.extend(rng.sample(population,min(per_class,len(population))))
    return ids

def manifests(job):
    train=[r for r in job['dataset']['records'] if r['split']=='train'];p=job['training_policy'];count=job['steps']*job['batch_size']*job['accumulation'];ids=sample_ids(train,p['sampler'],job['seed'],count)
    sample={'schema':'saal-admitted-sample-stream-v1','job_digest':sha(job),'mode':p['sampler'],'seed':job['seed'],'record_ids':ids,'unique_records':len(set(ids)),'sample_exposures':len(ids),'complete_training_coverage':set(ids)=={r['record_id'] for r in train},'completed_epochs':len(ids)//len(train) if p['sampler']=='complete-balanced-epochs-v1' else None,'trust':'Deterministic source reconstruction and worker-recorded stream, not hostile-worker attestation.'}
    diag={'schema':'saal-training-diagnostic-population-v1','job_digest':sha(job),'seed':p['diagnostic_seed'],'per_class':p['diagnostic_per_class'],'record_ids':diagnostic_ids(train,p['diagnostic_seed'],p['diagnostic_per_class']),'scope':'Resubstitution diagnostics; never checkpoint selection or qualification.'}
    return sample,diag
