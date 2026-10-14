"""Order-stable descriptive public effect reporting; no independence assertion."""
from .checkpoint_selection import paired_family_gain


def family_gain(before, after):
    def ordered(metrics):
        return dict(metrics, family_accuracy=dict(sorted(metrics['family_accuracy'].items())))
    return dict(paired_family_gain(ordered(before), ordered(after)),
        algorithm='sorted-family-bootstrap-v2', family_ids=sorted(before['family_accuracy']))
