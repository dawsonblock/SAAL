import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('v6_qa_result_tested', ROOT/'tools/adaptive_qa_result.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


@pytest.fixture
def recovered(tmp_path):
    result = {'returncodes': {n: 0 for n in qa.SUITES}, 'source_commit': 'a'*40,
        'input_archive_sha256': 'b'*64, 'hash_lock_enforced': True,
        'qualification_status': 'NOT_EVALUATED'}
    (tmp_path/'result.json').write_text(json.dumps(result))
    for n in qa.SUITES:
        (tmp_path/(n+'.xml')).write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="actual-case"/></testsuite></testsuites>')
    return tmp_path, result, {'phase': 'COMPLETED', 'cleanup_verified': True}


def test_all_eight_suites_and_teardown_required(recovered):
    root, _, state = recovered
    assert qa.validate_results(root, state, 'a'*40, 'b'*64)['passed'] == 8


@pytest.mark.parametrize('fault', ['missing_adaptive', 'wrong_source', 'wrong_payload',
    'no_cleanup', 'false_count', 'hidden_failure'])
def test_qa_cannot_claim_success_with_partial_or_contradictory_evidence(recovered, fault):
    root, result, state = recovered
    if fault == 'missing_adaptive':
        del result['returncodes']['adaptive']
    elif fault == 'wrong_source':
        result['source_commit'] = 'c'*40
    elif fault == 'wrong_payload':
        result['input_archive_sha256'] = 'd'*64
    elif fault == 'no_cleanup':
        state['cleanup_verified'] = False
    elif fault == 'false_count':
        (root/'adaptive.xml').write_text('<testsuite tests="999" failures="0" errors="0"><testcase/></testsuite>')
    else:
        (root/'adaptive.xml').write_text('<testsuite tests="1" failures="0" errors="0"><testcase><failure>hidden</failure></testcase></testsuite>')
    (root/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):
        qa.validate_results(root, state, 'a'*40, 'b'*64)
