"""Public probability calibration and selective prediction; no authority granted."""
import math
from .checkpoint_selection import metrics
from .pretrained_contract import DECISIONS
from .spec import sha


def validate_probabilities(records, outputs):
    if not records or len(records) != len(outputs):
        raise ValueError('complete probability inventory required')
    m = metrics(records, outputs, 'modernbert-seqcls-v1')
    if m['invalid_outputs']:
        raise ValueError('malformed probability vectors cannot be calibrated')
    return m


def temperature_output(output, temperature):
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not .25 <= temperature <= 4:
        raise ValueError('calibration temperature bounds')
    values = [math.log(max(x, 1e-12)) / temperature for x in output['probabilities']]
    offset = max(values); probabilities = [math.exp(x - offset) for x in values]
    total = sum(probabilities); probabilities = [x / total for x in probabilities]
    return {'decision': DECISIONS[max(range(4), key=lambda i: probabilities[i])], 'probabilities': probabilities}


def fit_temperature(records, outputs, *, model_digest):
    from .attempt_authority import digest
    digest(model_digest); validate_probabilities(records, outputs)
    if any(r['split'] != 'development' for r in records) or len({r['expected_decision'] for r in records}) != 4:
        raise ValueError('temperature fit requires four-class public development only')
    temperatures = [.25, .5, .75, 1., 1.5, 2., 3., 4.]
    def nll(t):
        return -sum(math.log(max(temperature_output(o, t)['probabilities'][DECISIONS.index(r['expected_decision'])], 1e-12)) for r, o in zip(records, outputs)) / len(records)
    t = min(temperatures, key=lambda x: (nll(x), abs(x - 1)))
    return {'schema': 'saal-public-temperature-fit-v1', 'temperature': t,
            'grid': temperatures, 'fit_records_digest': sha(records), 'fit_predictions_digest': sha(outputs),
            'model_digest': model_digest, 'nll': nll(t), 'scope': 'PUBLIC_DIAGNOSTIC_ONLY',
            'qualification_admission': False, 'external_deployment_allowed': False}


def selective_report(records, outputs, *, thresholds=(0., .5, .7, .9), temperature=1.):
    validate_probabilities(records, outputs)
    if any(r['split'] not in ('development', 'regression') for r in records):
        raise ValueError('selective diagnostics require public data')
    if not thresholds or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('registered threshold grid')
    calibrated = [temperature_output(o, temperature) for o in outputs]
    rows = []
    for threshold in thresholds:
        accepted = [(r, o) for r, o in zip(records, calibrated) if max(o['probabilities']) >= threshold]
        errors = sum(r['expected_decision'] != o['decision'] for r, o in accepted)
        rows.append({'threshold': threshold, 'coverage': len(accepted) / len(records),
                     'accepted': len(accepted), 'errors': errors,
                     'conditional_error': errors / len(accepted) if accepted else None,
                     'escalated': len(records) - len(accepted),
                     'correct_clarifications': sum(r['expected_decision'] == 'ASK' and o['decision'] == 'ASK' for r, o in accepted)})
    return {'schema': 'saal-public-selective-diagnostics-v1', 'uncalibrated': metrics(records, outputs, 'modernbert-seqcls-v1'),
            'calibrated': metrics(records, calibrated, 'modernbert-seqcls-v1'), 'selection_curve': rows,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
            'limits': 'Low-confidence escalation is not scored as correct ASK. Confidence does not establish correctness or authorization. Correlated public data give descriptive conditional errors only.'}
