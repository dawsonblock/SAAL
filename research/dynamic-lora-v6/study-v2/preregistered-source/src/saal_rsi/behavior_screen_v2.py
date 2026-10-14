"""Invalid scores invalidate a pair even when the textual decision is correct."""
from collections import defaultdict
from .behavior_screen import screen
from .behavior_screen import error_audit as previous_audit
from .epoch_metrics import prediction_equal


def error_audit(dataset, outputs, split):
    report = previous_audit(dataset, outputs, split)
    groups = defaultdict(list)
    for row, output in zip([r for r in dataset['records'] if r['split'] == split], outputs):
        valid = prediction_equal([output], [output], 'modernbert-seqcls-v1', 1e-4)
        groups[row['cluster_id']].append((row, output, valid))
    failed = []
    for cluster, group in groups.items():
        if len(group) == 2 and len({r['expected_decision'] for r, _, _ in group}) == 2:
            if not all(valid and output['decision'] == row['expected_decision'] for row, output, valid in group):
                failed.append({'cluster_id': cluster, 'record_ids': [r['record_id'] for r, _, _ in group]})
    return dict(report, schema='saal-public-error-audit-v2', failed_counterfactual_pairs=failed)
