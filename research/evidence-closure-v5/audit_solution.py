"""Separate raw model errors from the trusted-state planning control."""
from pathlib import Path
import json
import sys
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.structured_choice import plan
from saal_rsi.spec import sha, canonical


def run():
    dataset = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    rows = {r['record_id']: r for r in dataset['records']}
    states, sources = {}, {}
    for name, field in [('research/diverse-learning-v1/state-audit.json', 'records'),
                       ('research/learning-reliability-v2/corpus/state-audit.json', 'compact_states'),
                       ('research/behavior-closure-v4/state-audit.json', 'records')]:
        path = ROOT/name
        source = json.loads(path.read_text())
        sources[name] = sha(source)
        for item in source[field]:
            row = rows.get(item['record_id'])
            if row is None:
                continue
            if item['record_id'] in states or item['prompt_digest'] != sha(row['prompt']) or item['state_digest'] != sha(item['state']):
                raise ValueError('duplicate or substituted state provenance')
            result = plan(item['state'])
            if result['decision'] != row['expected_decision']:
                raise ValueError('structured control disagrees with label; do not silently relabel')
            states[item['record_id']] = result
    analysis = {}
    for model in ('full47', 'lora211'):
        for split in ('development', 'regression'):
            audit = json.loads((ROOT/f'research/behavior-closure-v4/failure-audits/v4-{model}-terminal-{split}.json').read_text())
            reasons = Counter()
            mapped = critical = very_confident = 0
            for error in audit['errors']:
                decision = states.get(error['record_id'])
                if decision is None:
                    continue
                mapped += 1
                critical += error['critical_action_proposal']
                very_confident += error['confidence'] is not None and error['confidence'] >= .95
                predicted = error['predicted']
                if predicted in ('CHEAP', 'FAST') and predicted not in decision['known_feasible_options']:
                    reason = 'PROPOSED_KNOWN_INFEASIBLE_OPTION'
                elif decision['reason'] != 'CURRENT_PRIORITY_OVER_FEASIBLE_ACTIONS':
                    reason = decision['reason']
                elif predicted == 'REFUSE':
                    reason = 'UNNECESSARY_REFUSAL'
                elif predicted == 'ASK':
                    reason = 'UNNECESSARY_CLARIFICATION'
                else:
                    reason = 'WRONG_CURRENT_OBJECTIVE'
                reasons[reason] += 1
            analysis[model+'_'+split] = {'raw_errors': len(audit['errors']),
                'state_mapped_errors': mapped, 'unmapped_errors': len(audit['errors'])-mapped,
                'mapped_critical_proposals': critical, 'wrong_predictions_with_softmax_at_least_95_percent': very_confident,
                'mapped_failure_categories': dict(reasons), 'raw_metrics': audit['metrics']}
    report = {'schema': 'saal-structured-planning-control-analysis-v1', 'dataset_digest': sha(dataset),
        'state_source_digests': sources, 'state_covered_records': len(states),
        'state_uncovered_records': len(rows)-len(states), 'control_label_disagreements': 0,
        'historical_neural_predictions_changed': False, 'historical_model_analysis': analysis,
        'limits': 'Exact state planning is given authoritative structured facts. It does not parse prompts and is not comparable to raw neural accuracy. Shared authors/oracles and missing human review remain. Mechanical consistency is not semantic or independent qualification.',
        'human_reviews': 0, 'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    with (PROJECT/'SOLUTION_ANALYSIS.json').open('xb') as out: out.write(canonical(report))
    print(json.dumps({'state_covered_records': len(states), 'mapped_failure_counts':
        {k:v['mapped_failure_categories'] for k,v in analysis.items()}}))


if __name__ == '__main__': run()
