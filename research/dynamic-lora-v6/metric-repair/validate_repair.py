"""Local QA only: no signing, training admission, cloud or budget mutation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SUITES = (
    ('core', 'tests', 'src'),
    ('corpus', 'reviewed-decision-corpus/tests', 'src:reviewed-decision-corpus/src'),
    ('reporting', 'research/reliable-learning-v3/tests', 'src'),
    ('project', 'research/behavior-closure-v4/tests', 'src'),
    ('diverse', 'research/diverse-learning-v1/tests', 'src:research/diverse-learning-v1/src'),
    ('reliability', 'research/learning-reliability-v2/tools/test_postprocessing.py', 'src'),
    ('repairs', 'research/evidence-closure-v5/tests', 'src'),
    ('adaptive', 'research/dynamic-lora-v6/tests', 'src'),
    ('numerical_repair', 'research/dynamic-lora-v6/metric-repair/tests', 'src'),
)


def run(destination):
    destination = Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    tested_paths = list((ROOT/'src').rglob('*.py'))
    tested_paths += list(Path(__file__).parent.glob('*.py'))
    tested_paths += list((Path(__file__).parent/'tests').glob('*.py'))
    tested_paths += [Path(__file__).with_name('CAMPAIGN_DRAFT.json')]
    source_files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tested_paths)}
    results = []
    for name, target, paths in SUITES:
        started = time.monotonic()
        xml = destination/(name+'.xml')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in paths.split(':')))
        with (destination/(name+'.log')).open('xb') as log:
            result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p',
                'no:cacheprovider', target, '--junitxml='+str(xml)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        tree = ET.parse(xml).getroot()
        rows = list(tree) if tree.tag == 'testsuites' else [tree]
        counts = {k: sum(int(r.get(k, 0)) for r in rows) for k in ('tests', 'failures', 'errors', 'skipped')}
        cases = list(tree.iter('testcase'))
        if len(cases) != counts['tests']:
            raise ValueError('JUnit collection mismatch')
        row = {'suite': name, 'returncode': result.returncode, **counts,
               'wall_seconds': time.monotonic()-started}
        results.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode or counts['failures'] or counts['errors']:
            break
    from saal_rsi.adaptive_contract_v3 import worker_identity
    from saal_rsi.numerical_metrics_v1 import POLICY_DIGEST
    source_unchanged = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                           for name, digest in source_files.items())
    value = {'schema': 'saal-numerical-repair-local-qa-v1',
        'status': 'PASSED' if source_unchanged and len(results) == len(SUITES) and all(
            not r['returncode'] and not r['failures'] and not r['errors'] for r in results) else 'FAILED',
        'suites': results, 'totals': {k: sum(r[k] for r in results)
            for k in ('tests', 'failures', 'errors', 'skipped')},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'executable': sys.executable},
        'worker_identity': worker_identity(), 'numerical_metric_policy_digest': POLICY_DIGEST,
        'tested_source_files': source_files, 'tested_source_unchanged': source_unchanged,
        'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'evidence_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.iterdir()) if p.is_file()},
        'fresh_colab_validation': 'NOT_RUN', 'new_campaign_authorized': False,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    with (destination/'RESULT.json').open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')
    if value['status'] != 'PASSED':
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: validate_repair.py NEW_EVIDENCE_DIRECTORY')
    sys.path.insert(0, str(ROOT/'src'))
    run(sys.argv[1])
