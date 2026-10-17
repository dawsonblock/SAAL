"""Summarize the frozen public A100 diagnostic without qualification claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))

from saal_rsi.adaptive_artifacts_v4 import validate_bundle,read_json
from saal_rsi.adaptive_evaluator_v4 import validate_jobs,validate_sources
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical,parse_unique_json,sha


def _compact(metrics):
    critical=metrics['critical_errors']['counts']
    return {'accuracy':metrics['accuracy'],
        'counterfactual_pairs_correct':metrics['counterfactual_pair_correct'],
        'counterfactual_pairs_total':metrics['counterfactual_pair_total'],
        'complete_pair_accuracy':metrics['complete_pair_accuracy'],
        'worst_family_accuracy':metrics['worst_family_accuracy'],
        'family_macro_accuracy':metrics['family_macro_accuracy'],
        'unauthorized_action_count':critical['authorization_violation']+
            critical['unsafe_action_when_clarification_required'],
        'refuse_to_action':critical['authorization_violation'],
        'ask_to_action':critical['unsafe_action_when_clarification_required'],
        'false_refusal':critical['false_refusal'],
        'class_accuracy':metrics['class_accuracy']}


def _run_diagnostics(root, result):
    rows=[parse_unique_json(line) for line in (root/'training_metrics.jsonl').read_bytes().splitlines()]
    progress=[r for r in rows if r.get('monitor')=='training_progress']
    checkpoints=[r for r in rows if r.get('monitor')=='checkpoint_metrics']
    rates={json.dumps(r['learning_rates'],sort_keys=True) for r in progress}
    losses=[r['cross_entropy'] for r in progress]
    norms=[r['gradient_norm'] for r in progress]
    ranks=[r['active_rank'] for r in progress]
    rank_rows=[parse_unique_json(line) for line in (root/'rank_history.jsonl').read_bytes().splitlines()]
    return {'completed_updates':len(progress),
        'first_logged_learning_rates':progress[0]['learning_rates'] if progress else None,
        'unique_learning_rate_records':sorted(rates),
        'cross_entropy_first':losses[0] if losses else None,
        'cross_entropy_min':min(losses) if losses else None,
        'cross_entropy_last':losses[-1] if losses else None,
        'gradient_norm_min':min(norms) if norms else None,
        'gradient_norm_max':max(norms) if norms else None,
        'active_rank_initial':rank_rows[0]['aggregate_active_rank'] if rank_rows else None,
        'active_rank_terminal':rank_rows[-1]['aggregate_active_rank'] if rank_rows else None,
        'active_rank_by_layer_terminal':rank_rows[-1]['active_rank_by_layer'] if rank_rows else None,
        'checkpoint_curve':[{'step':r['step'],**_compact(r['development'])} for r in checkpoints],
        'result_learning_status':result.get('learning_status'),
        'worker_seconds':result.get('total_worker_seconds'),
        'resource_accounting':result.get('resource_accounting')}


def analyze(admission,campaign):
    admission=Path(admission).resolve();campaign=Path(campaign).resolve()
    pin=(admission/'controller.pub').read_text().strip()
    protocol_envelope=parse_unique_json((admission/'signed_protocol.json').read_bytes())
    protocol=verify(protocol_envelope,pin)
    validate_sources(protocol,ROOT)
    job_envelopes=[parse_unique_json(p.read_bytes()) for p in sorted((admission/'jobs').glob('*.json'))]
    jobs=validate_jobs(protocol,job_envelopes,pin)
    runs=[]
    for descriptor,job in zip(protocol['runs'],jobs):
        outcome_path=campaign/'outcomes.json'
        outcomes=parse_unique_json(outcome_path.read_bytes())['runs'] if outcome_path.exists() else []
        outcome=next((row for row in outcomes if row.get('job_digest')==descriptor['job_digest']),None)
        record={'name':descriptor['name'],'method':descriptor['method'],'seed':descriptor['seed'],
            'job_digest':descriptor['job_digest'],'attempt_state':outcome['state'] if outcome else 'NOT_STARTED'}
        if outcome and outcome['state']=='COMPLETED':
            bundle=Path(outcome['artifact_dir'])
            result=validate_bundle(bundle,job,pin,require_replay=True)
            record.update({'artifact_inventory_digest':sha(read_json(bundle/'artifact_inventory.json')),
                'snapshots':{role:_compact(result['snapshots'][role]['metrics']) for role in ('initial','best','terminal')},
                'snapshot_steps':{role:result['snapshots'][role]['step'] for role in ('initial','best','terminal')},
                'diagnostics':_run_diagnostics(bundle,result),
                'rank_allocation':result['allocation'],
                'local_replay_verified':outcome.get('local_replay_verified') is True,
                'artifact_archive_sha256':outcome['artifact_archive_sha256']})
        elif outcome:
            record['failure']=outcome.get('error')
        runs.append(record)
    completed=[r for r in runs if r['attempt_state']=='COMPLETED']
    base_hashes=[]
    for item in completed:
        descriptor=next(r for r in protocol['runs'] if r['job_digest']==item['job_digest'])
        job=next(j for j in jobs if sha(j)==descriptor['job_digest'])
        outcome=next(r for r in parse_unique_json((campaign/'outcomes.json').read_bytes())['runs']
                     if r['job_digest']==item['job_digest'])
        result=validate_bundle(Path(outcome['artifact_dir']),job,pin,require_replay=True)
        base_hashes.append(result['unadapted_model_hash'])
    terminal=parse_unique_json((campaign/'campaign-terminal.json').read_bytes()) if (campaign/'campaign-terminal.json').exists() else None
    value={'schema':'saal-a100-engineering-diagnostic-report-v1',
        'campaign_id':protocol['budget']['campaign_id'],'protocol_digest':sha(protocol),
        'signer_status':protocol['signer_status'],'data_review_status':protocol['data_review_status'],
        'scope':protocol['scope'],'runtime_requirements':protocol['runtime_requirements'],
        'seed':47,'methods':list(r['method'] for r in protocol['runs']),
        'all_runs_completed':len(completed)==len(protocol['runs']),
        'all_runs_independently_replayed':len(completed)==len(protocol['runs']) and all(r.get('local_replay_verified') for r in completed),
        'same_unadapted_model_hash_across_completed_runs':len(set(base_hashes))<=1,
        'runs':runs,'campaign_terminal':terminal,
        'interpretation':'One-seed public engineering diagnostic only. It can localize training behavior; it cannot establish independent generalization, retention, causal mechanism attribution, or qualification.',
        'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED',
        'retention':'NOT_MEASURED','operational_custody_independence':'NOT_ESTABLISHED',
        'external_deployment_allowed':False}
    (campaign/'A100_DIAGNOSTIC_REPORT.json').write_bytes(canonical(value))
    lines=['# A100 training-diagnostic pilot','',
        '**Scope:** one seed, public train/development diagnostics; no protected evaluation.',
        '**Signer:** self-signed local development key; independent trust is not established.',
        '**Data:** oracle-checked, not independently reviewed. Results are not qualification.', '',
        '| Method | State | Initial accuracy / pairs | Best accuracy / pairs | Terminal accuracy / pairs | Unauthorized actions (terminal) | Selected step |',
        '|---|---|---:|---:|---:|---:|---:|']
    for run in runs:
        if 'snapshots' not in run:
            lines.append(f"| {run['method']} | {run['attempt_state']} | — | — | — | — | — |")
            continue
        initial,best,terminal_m=(run['snapshots'][k] for k in ('initial','best','terminal'))
        lines.append(f"| {run['method']} | {run['attempt_state']} | {initial['accuracy']:.3f} / {initial['counterfactual_pairs_correct']}/{initial['counterfactual_pairs_total']} | {best['accuracy']:.3f} / {best['counterfactual_pairs_correct']}/{best['counterfactual_pairs_total']} | {terminal_m['accuracy']:.3f} / {terminal_m['counterfactual_pairs_correct']}/{terminal_m['counterfactual_pairs_total']} | {terminal_m['unauthorized_action_count']} | {run['snapshot_steps']['best']} |")
    lines += ['', '## What this can answer', '',
        'The logs compare initial, best-development and terminal checkpoints, show the loss and effective learning rates, track gradient norms and rank allocations, and expose the first checkpoint where critical errors appear. The selector labels a run `NO_QUALIFYING_TRAINED_CHECKPOINT` if no positive-step snapshot both improves accuracy and counterfactual pairs and passes the public critical-error/family guardrails. Step zero is retained only as the initial reference.', '',
        '## Limits', '',
        'All runs share one seed and the same public synthetic task population. This is a diagnostic replication, not an independent or human-reviewed evaluation. It does not measure retention or prove that adaptive rank allocation caused an outcome. No result authorizes promotion or external deployment.', '',
        f"Campaign terminal status: `{terminal.get('campaign_state') if terminal else 'UNAVAILABLE'}`.",
        'Qualification: `NOT_EVALUATED`. Independent generalization: `NOT_ESTABLISHED`. Retention: `NOT_MEASURED`.']
    (campaign/'A100_DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--admission',required=True)
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    result=analyze(args.admission,args.campaign)
    print(json.dumps({'runs_completed':sum(r['attempt_state']=='COMPLETED' for r in result['runs']),
        'run_count':len(result['runs']),'all_replayed':result['all_runs_independently_replayed'],
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}))
