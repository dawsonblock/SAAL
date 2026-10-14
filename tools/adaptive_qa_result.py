"""Validate complete v6 QA recovery, including the newly added adapter suite."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

SUITES = ('core', 'corpus', 'reporting', 'project', 'diverse', 'reliability', 'repairs', 'adaptive')


def validate_results(evidence, state, source_commit, input_sha256):
    evidence = Path(evidence)
    result = json.loads((evidence/'result.json').read_text())
    if (result.get('returncodes') != {n: 0 for n in SUITES}
            or any(type(v) is not int for v in result['returncodes'].values())
            or result.get('source_commit') != source_commit
            or result.get('input_archive_sha256') != input_sha256
            or result.get('hash_lock_enforced') is not True
            or result.get('qualification_status') != 'NOT_EVALUATED'
            or state.get('cleanup_verified') is not True or state.get('phase') != 'COMPLETED'):
        raise ValueError('complete source-bound eight-suite QA and verified teardown required')
    totals = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
    suites = []
    for name in SUITES:
        tree = ET.parse(evidence/(name+'.xml')).getroot()
        rows = list(tree) if tree.tag == 'testsuites' else [tree]
        counts = {key: sum(int(r.attrib.get(key, '0')) for r in rows) for key in totals}
        if counts['tests'] <= 0 or counts['failures'] or counts['errors']:
            raise ValueError('JUnit does not confirm passing suite '+name)
        cases = list(tree.iter('testcase'))
        if len(cases) != counts['tests'] or sum(1 for c in cases if c.find('skipped') is not None) != counts['skipped']:
            raise ValueError('JUnit count mismatch '+name)
        if any(c.find('failure') is not None or c.find('error') is not None for c in cases):
            raise ValueError('JUnit failure contradicts counters '+name)
        suites.append({'suite': name, **counts})
        for key in totals:
            totals[key] += counts[key]
    return {**result, 'suites': suites, 'totals': totals,
            'passed': totals['tests']-totals['skipped'], 'cleanup_verified': True}
