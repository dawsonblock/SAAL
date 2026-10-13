"""Two-seed matched public pilot. New source/attempts, no historical rewriting."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
sys.path.append(str(ROOT/'research/diverse-learning-v1/src'))
from saal_rsi.spec import canonical, sha
from saal_rsi.model_contract import sign, verify
from saal_rsi.attempt_service import private_key
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.epoch_contract import new_job, SOURCES, worker_identity
from saal_rsi.epoch_backend_v6 import execute_campaign, require_storage
from saal_rsi.compositional_corpus_v5 import validate

RUNS = [(f'{condition}_{seed}', condition, seed) for seed in (47, 83)
        for condition in ('parent_control', 'compositional')]+[('compositional_full_47', 'compositional_full', 47)]


def source_map():
    names = ['src/saal_rsi/'+name for name in SOURCES]
    names += ['src/saal_rsi/'+name for name in ('epoch_backend_v6.py', 'epoch_artifacts_v3.py',
        'epoch_artifacts_v2.py', 'model_monitor_v3.py', 'model_monitor.py', 'epoch_registry.py',
        'compositional_corpus_v5.py', 'choice_policy_v1.py', 'structured_choice.py', 'family_gain_v2.py')]
    names += ['research/evidence-closure-v5/campaign.py',
        'research/diverse-learning-v1/src/saal_diverse/dataset.py',
        'research/diverse-learning-v1/src/saal_diverse/pair_recipe.py',
        'research/diverse-learning-v1/src/saal_diverse/policy.py',
        'configs/dependency-locks/pretrained-linux-x86_64-py313.txt']
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def prepare(key_path):
    parent = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    data = json.loads((PROJECT/'dataset.json').read_text())
    audit = json.loads((PROJECT/'state-audit.json').read_text())
    validate(data, audit, parent)
    old_ids = {row['record_id'] for row in parent['records']}
    control = dict(data, records=[r for r in data['records'] if r['split'] != 'train' or r['record_id'] in old_ids],
        sampling_description='Matched parent training control; exact same exposed expanded development/regression populations as compositional treatment. No independent review or protected data.')
    out = PROJECT/'experiment'
    out.mkdir(exist_ok=False)
    key = private_key(key_path)
    pin = key.public_key().public_bytes_raw().hex()
    jobs = []
    for name, condition, seed in RUNS:
        method = 'full' if condition == 'compositional_full' else 'lora'
        spec = model_spec('answerdotai/ModernBERT-base', method=method, rank=4,
            backend='colab', precision='float32', budget='closure5_'+name, max_tokens=256)
        job = new_job(spec, control if condition == 'parent_control' else data,
            seed=seed, steps=1000, batch_size=8, accumulation=2,
            learning_rate=.0003, head_learning_rate=.0003,
            encoder_learning_rate=.00001 if method == 'full' else .0003,
            head_warmup_steps=100 if method == 'full' else 0,
            max_seconds=2400, class_balance=True, checkpoint_interval=100)
        require_storage(PROJECT, job)
        (out/(name+'-job.json')).write_bytes(canonical(sign(job, key)))
        jobs.append({'name': name, 'condition': condition, 'seed': seed, 'job_digest': sha(job)})
    protocol = {'schema': 'saal-public-evidence-closure-pilot-v1', 'source_files': source_map(),
        'worker_identity': worker_identity(), 'dataset_digest': sha(data), 'parent_dataset_digest': sha(parent),
        'state_audit_digest': sha(audit), 'runs': jobs,
        'hypothesis': 'Balanced current/archived-objective transitions and wider numeric supervision may reduce observed correction and infeasibility errors.',
        'comparison': 'Two matched seeds, same initialization, fixed-rank4 LoRA, steps, head/adapter rates, loss, sampler and 16000 sample exposures. Only training population changes. Both use the exact same expanded public diagnostics.',
        'exploratory_full_tuning': 'One separately declared full-tuning seed47, 100-step head warmup included in 1000 total updates; encoder learning rate 1e-5. Compare old/new public strata with the preserved earlier full47 model. Not a five-seed full-tuning comparison.',
        'selection': 'Unchanged safety-first development selector; retain initial/best/terminal. Public regression never selects weights.',
        'primary_report': 'Terminal measurements on the exact old public population, newly authored public population, and combined population, separately. Report every seed and failure; no best-seed final evaluation.',
        'limitations': ['two-seed exploratory pilot', 'fewer exposures per old training example in expanded population',
            'failure-informed shared author/generator/oracle', 'zero human reviews', 'no protected final evaluation'],
        'screen': {'complete_pairs': .9, 'worst_family_accuracy': .8, 'critical_proposals': 0,
                   'invalid_outputs': 0, 'four_class_coverage': True},
        'budget': {'jobs': 5, 'per_job_seconds': 2400, 'lifecycle_seconds': 20200,
                   'automatic_training_retries': 0, 'financial_cap': False},
        'probability_replay_tolerance': 1e-4, 'protected_alpha_consumed': False,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    (out/'controller.pub').write_text(pin+'\n')
    (out/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    print(json.dumps({'registered_jobs': len(jobs), 'source_digest': sha(source_map()),
        'data_digest': sha(data), 'qualification_status': 'NOT_EVALUATED'}))


def run():
    out = PROJECT/'experiment'
    pin = (out/'controller.pub').read_text().strip()
    protocol = verify(json.loads((out/'signed_protocol.json').read_text()), pin)
    if protocol['source_files'] != source_map() or protocol['worker_identity'] != worker_identity():
        raise ValueError('preregistered source changed')
    jobs = [json.loads((out/(row['name']+'-job.json')).read_text()) for row in protocol['runs']]
    if [sha(verify(job, pin)) for job in jobs] != [row['job_digest'] for row in protocol['runs']]:
        raise ValueError('signed job substitution')
    return execute_campaign(out/'colab', jobs, pin, out/'artifacts', out/'budget.sqlite3',
                            max_seconds=protocol['budget']['lifecycle_seconds'])


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare(sys.argv[2])
    elif sys.argv[1] == 'run': run()
    else: raise ValueError('prepare or run required')
