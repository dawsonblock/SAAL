import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest
from saal_rsi import model_monitor_v3 as monitor

SESSION = 'saal-'+'a'*20


def plan():
    return {'session': SESSION, 'commands': [['colab', 'new', '-s', SESSION],
        ['colab', 'stop', '-s', SESSION]], 'phases': ['CREATE'],
        'estimated_max_seconds': 100, 'job_digest': 'sha256:'+'b'*64}


class Guard:
    pid = 123
    def wait(self, timeout):
        return 0


@pytest.mark.parametrize('fault', ['phases', 'session', 'cleanup', 'cross_session', 'deadline', 'command'])
def test_invalid_plan_has_no_watchdog_or_cloud_side_effect(tmp_path, monkeypatch, fault):
    value = plan()
    if fault == 'phases': value['phases'] = []
    elif fault == 'session': value['session'] = '../other'
    elif fault == 'cleanup': value['commands'][-1] = ['colab', 'stop', '-s', 'other']
    elif fault == 'cross_session': value['commands'][0][-1] = 'other'
    elif fault == 'deadline': value['estimated_max_seconds'] = float('nan')
    else: value['commands'][0] = 'colab new'
    def forbidden(*args): raise AssertionError('must validate before side effects')
    monkeypatch.setattr(monitor, 'launch_watchdog', forbidden)
    monkeypatch.setattr(monitor, 'stream_command', forbidden)
    with pytest.raises(ValueError): monitor.execute_colab_plan(value, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('failure', ['state', 'events', 'child_and_state', 'cleanup_logs'])
def test_evidence_io_failure_never_prevents_stop_and_status(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(monitor, 'launch_watchdog', lambda *args: Guard())
    original = monitor.write_json
    original_open = Path.open
    def write(path, value):
        if failure in ('state', 'child_and_state') and calls:
            raise OSError(errno.ENOSPC, 'full disk')
        return original(path, value)
    monkeypatch.setattr(monitor, 'write_json', write)
    def opened(path, *args, **kwargs):
        if failure == 'events' and path.name == 'events.jsonl':
            raise OSError(errno.ENOSPC, 'full journal')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    def command(args, *extra):
        calls.append(args[1])
        if args[1] == 'new' and failure == 'child_and_state':
            raise RuntimeError('original child failure')
        if failure == 'cleanup_logs' and args[1] in ('stop', 'status'):
            raise OSError(errno.ENOSPC, 'cleanup log unavailable')
        return f"Session '{SESSION}' not found." if args[1] == 'status' else ''
    monkeypatch.setattr(monitor, 'stream_command', command)
    fallback = []
    def memory_command(args, **kwargs):
        fallback.append(args)
        return subprocess.CompletedProcess(args, 0,
            f"Session '{SESSION}' not found." if args[1] == 'status' else '', '')
    monkeypatch.setattr(monitor.subprocess, 'run', memory_command)
    expected = RuntimeError if failure == 'child_and_state' else monitor.EvidenceWriteError
    with pytest.raises(expected) as error: monitor.execute_colab_plan(plan(), tmp_path)
    if failure == 'events':
        assert calls == []  # Initial event failure prevents provisioning.
    else:
        assert calls[-2:] == ['stop', 'status']
    if failure == 'child_and_state': assert str(error.value) == 'original child failure'
    if failure == 'cleanup_logs':
        assert [p[1] for p in fallback] == ['stop', 'status']
        assert json.loads((tmp_path/'monitor-state.json').read_text())['cleanup_verified'] is True


def test_stop_timeout_still_queries_status_and_does_not_invent_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'launch_watchdog', lambda *args: Guard())
    calls = []
    def command(args, *extra):
        calls.append(args[1])
        if args[1] == 'stop': raise TimeoutError('uncertain stop')
        return 'still running' if args[1] == 'status' else ''
    monkeypatch.setattr(monitor, 'stream_command', command)
    with pytest.raises(RuntimeError, match='unresolved'): monitor.execute_colab_plan(plan(), tmp_path)
    assert calls == ['new', 'stop', 'status']
    state = json.loads((tmp_path/'monitor-state.json').read_text())
    assert state['terminal'] and not state['cleanup_verified'] and state['phase'] == 'FAILED'


@pytest.mark.parametrize('state', ['missing', 'corrupt', 'substituted'])
def test_real_watchdog_child_uses_launch_identity_when_state_is_unavailable(tmp_path, state):
    path = tmp_path/'state.json'
    if state == 'corrupt': path.write_text('{bad')
    elif state == 'substituted': path.write_text(json.dumps({'session': 'other', 'terminal': True, 'cleanup_verified': True}))
    binary = tmp_path/'colab'
    binary.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\np=Path('+repr(str(tmp_path/'calls'))+')\nwith p.open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif sys.argv[1]=="status":print("Session "+repr(sys.argv[-1])+" not found.")\n')
    binary.chmod(0o700)
    subprocess.run([sys.executable, '-I', str(Path(monitor.__file__).resolve()), '--state', str(path),
        '--deadline', str(time.time()-1), '--session', SESSION],
        env=dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH']),
        check=True, capture_output=True, timeout=15)
    calls = [json.loads(line) for line in (tmp_path/'calls').read_text().splitlines()]
    assert calls == [['stop', '-s', SESSION], ['status', '-s', SESSION]]
    receipt = json.loads((tmp_path/'watchdog-result.json').read_text())
    assert receipt['absence_confirmed'] is True and receipt['session'] == SESSION
