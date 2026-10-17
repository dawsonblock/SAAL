"""Fail-closed account-level Colab compute-unit guard for v4 pilots."""
import re


def parse_usage(text):
    if type(text) is not str or len(text) > 20_000:
        raise ValueError('bounded Colab usage response required')
    values = {}
    patterns = {
        'balance': r'^Current balance: ([0-9]+(?:\.[0-9]+)?) compute units$',
        'rate': r'^Usage rate: ([0-9]+(?:\.[0-9]+)?)/hr$',
        'active_assignments': r'^Active assignments: ([0-9]+)$',
    }
    for line in text.splitlines():
        for key, pattern in patterns.items():
            match = re.fullmatch(pattern, line.strip())
            if match:
                if key in values:
                    raise ValueError('duplicate Colab usage field')
                values[key] = float(match.group(1)) if key != 'active_assignments' else int(match.group(1))
    if set(values) != set(patterns):
        raise ValueError('Colab usage response lacks balance, rate or assignment count')
    if values['balance'] < 0 or values['rate'] < 0 or values['active_assignments'] < 0:
        raise ValueError('negative Colab usage value')
    return values


def assess_budget(*, starting_balance, usage, cap, remaining_seconds,
                  require_single_assignment=False):
    """Return actual spend and conservative projected spend under the signed cap."""
    if (type(starting_balance) not in (int, float) or starting_balance < 0
            or type(cap) is not int or not 1 <= cap <= 100
            or type(remaining_seconds) not in (int, float) or remaining_seconds < 0
            or type(usage) is not dict
            or set(usage) != {'balance', 'rate', 'active_assignments'}):
        raise ValueError('valid signed Colab compute budget inputs required')
    if type(usage['balance']) not in (int, float) or type(usage['rate']) not in (int, float):
        raise ValueError('numeric Colab usage values required')
    if type(usage['active_assignments']) is not int or usage['active_assignments'] < 0:
        raise ValueError('valid active assignment count required')
    spent = float(starting_balance) - float(usage['balance'])
    if spent < -0.01:
        raise ValueError('account balance increased during admitted campaign')
    spent = max(0.0, spent)
    projected = spent + float(usage['rate']) * float(remaining_seconds) / 3600.0
    if spent > cap or projected > cap:
        raise ValueError('signed compute-unit cap would be exceeded')
    if require_single_assignment and usage['active_assignments'] != 1:
        raise ValueError('Colab assignment count is ambiguous for this campaign')
    return {'reported_spend_compute_units': round(spent, 6),
            'projected_spend_compute_units': round(projected, 6),
            'usage_rate_per_hour': float(usage['rate']),
            'active_assignments': usage['active_assignments'],
            'within_signed_cap': True}
