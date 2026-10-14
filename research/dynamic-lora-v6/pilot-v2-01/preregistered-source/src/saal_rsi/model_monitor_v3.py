"""Evidence failure must stop work, but must never prevent runtime teardown.

Versioned replacement for new controller runs. Historical monitoring records and
source remain unchanged. This is lifecycle supervision, not worker containment.
"""
import datetime
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from saal_rsi.model_monitor import stream_command, write_json, redact_monitor_text
else:
    from .model_monitor import stream_command, write_json, redact_monitor_text


class EvidenceWriteError(RuntimeError):
    pass


def validate_plan(plan):
    if type(plan) is not dict:
        raise ValueError('plan object required')
    session = plan.get('session')
    if type(session) is not str or not re.fullmatch(r'saal-[a-f0-9]{20}', session):
        raise ValueError('exact session identity required')
    commands, phases = plan.get('commands'), plan.get('phases')
    if (type(commands) is not list or len(commands) < 2 or type(phases) is not list
            or len(phases) != len(commands)-1 or any(type(p) is not str or not p for p in phases)):
        raise ValueError('phase/command inventory')
    for command in commands:
        if type(command) is not list or len(command) < 2 or any(type(p) is not str or not p for p in command):
            raise ValueError('argument-vector commands required')
        if command[0] == 'colab' and ('-s' not in command or command[command.index('-s')+1:command.index('-s')+2] != [session]):
            raise ValueError('cross-session command denied')
    if commands[-1] != ['colab', 'stop', '-s', session]:
        raise ValueError('mandatory exact-session stop required')
    budget = plan.get('estimated_max_seconds')
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 1 <= budget <= 100000:
        raise ValueError('bounded lifecycle deadline')
    if type(plan.get('job_digest')) is not str:
        raise ValueError('job identity required')
    return plan


def launch_watchdog(state_path, deadline, session):
    with (state_path.parent/'watchdog.log').open('xb') as output:
        return subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()),
            '--state', str(state_path), '--deadline', str(deadline), '--session', session],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


def execute_colab_plan(plan, workspace):
    validate_plan(plan)  # Before creating a watchdog or provisioning any runtime.
    workspace = Path(workspace).resolve()
    state_path = workspace/'monitor-state.json'
    lock = threading.RLock()
    done = threading.Event()
    started = time.monotonic()
    deadline = time.time()+plan['estimated_max_seconds']
    state = {'schema': 'saal-colab-monitor-v3', 'session': plan['session'],
        'phase': 'PREPARED', 'job_digest': plan['job_digest'], 'deadline_epoch': deadline,
        'heartbeat_epoch': time.time(), 'terminal': False, 'cleanup_verified': False,
        'progress_step': 0, 'total_steps': 0, 'external_deployment_allowed': False,
        'evidence_errors': []}

    def record_failure(stage, exc):
        error = {'stage': stage, 'type': type(exc).__name__, 'detail': redact_monitor_text(str(exc))}
        if len(state['evidence_errors']) < 16:
            state['evidence_errors'].append(error)
            print(json.dumps({'monitor': 'evidence_write_failure', **error}), flush=True)

    def publish():
        with lock:
            state['heartbeat_epoch'] = time.time()
            state['elapsed_seconds'] = time.monotonic()-started
            try:
                write_json(state_path, state)
            except OSError as exc:
                record_failure('state', exc)

    def event(kind, **value):
        with lock:
            record = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'kind': kind, **value}
            try:
                with (workspace/'events.jsonl').open('a') as output:
                    output.write(json.dumps(record, allow_nan=False)+'\n')
            except OSError as exc:
                record_failure('events', exc)
            if kind == 'remote_monitor':
                state['last_remote_event'] = value
                if value.get('monitor') == 'campaign_progress':
                    for key in ('completed_runs', 'total_runs'):
                        state[key] = value[key]
                    state['current_seed'], state['current_mode'] = value['seed'], value['mode']
                if value.get('monitor') == 'training_progress':
                    state['progress_step'], state['total_steps'] = value['completed_steps'], value['total_steps']
                    state['latest_objective_loss'] = value['objective_loss']
            publish()

    def heartbeat():
        while not done.wait(5):
            publish()

    def require_evidence():
        if state['evidence_errors']:
            raise EvidenceWriteError('required monitoring evidence could not be persisted')

    def remote_event(row):
        event('remote_monitor', **row)
        require_evidence()  # Abort the active stream if monitoring storage fails.

    def cleanup_command(command, log, timeout):
        try:
            return stream_command(command, log, timeout)
        except OSError as exc:
            # Teardown must not require free disk space. Stop/status are bounded
            # trusted CLI operations; this fallback records the evidence loss.
            record_failure('cleanup_log', exc)
            reply = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, timeout=timeout)
            if len(reply.stdout)+len(reply.stderr) > 100000:
                raise ValueError('cleanup output bound')
            if reply.returncode:
                raise RuntimeError('cleanup CLI failed: '+redact_monitor_text(reply.stderr)[-1200:])
            return redact_monitor_text(reply.stdout)

    publish()
    require_evidence()  # No cloud work if initial local evidence cannot be saved.
    guard = launch_watchdog(state_path, deadline+10, plan['session'])
    state['watchdog_pid'] = guard.pid
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    attempted = False
    primary = cleanup_error = None
    try:
        for number, (command, phase) in enumerate(zip(plan['commands'][:-1], plan['phases'])):
            require_evidence()
            state['phase'], state['active_command'] = phase, command[1]
            event('command_started', phase=phase)
            require_evidence()
            attempted = True  # A timed-out create request may have created the runtime.
            stream_command(command, workspace/(str(number)+'-'+phase.lower()+'.log'),
                deadline-time.time(), remote_event)
            event('command_finished', phase=phase)
            require_evidence()
    except BaseException as exc:
        primary = exc
        state['error'] = {'type': type(exc).__name__, 'detail': redact_monitor_text(str(exc))}
        event('run_failed', error=state['error'])
    finally:
        if attempted:
            state['phase'] = 'CLEANUP'
            event('cleanup_started')  # A failed journal write does not skip either command.
            try:
                cleanup_command(plan['commands'][-1], workspace/'cleanup-stop.log', 60)
            except BaseException as exc:
                cleanup_error = exc
                state['stop_error'] = {'type': type(exc).__name__, 'detail': redact_monitor_text(str(exc))}
            # A failed stop transport must still be followed by a status query.
            try:
                confirmation = cleanup_command(['colab', 'status', '-s', plan['session']],
                    workspace/'cleanup-status.log', 30)
                if "Session '"+plan['session']+"' not found." not in confirmation:
                    raise RuntimeError('Runtime absence not confirmed')
                state['cleanup_verified'] = True
                cleanup_error = None  # Actual absence resolves an uncertain stop response.
                event('cleanup_confirmed')
            except BaseException as exc:
                cleanup_error = exc
                state['cleanup_error'] = {'type': type(exc).__name__, 'detail': redact_monitor_text(str(exc))}
                event('cleanup_failed', error=state['cleanup_error'])
        state['active_command'] = None
        state['terminal'] = True
        state['phase'] = 'FAILED' if primary or cleanup_error or state['evidence_errors'] else 'COMPLETED'
        done.set()
        thread.join(timeout=1)
        publish()
        if state['evidence_errors']:
            state['phase'] = 'FAILED'
            publish()  # Retain a failure state if storage has recovered.
        if not attempted or state['cleanup_verified']:
            try:
                guard.wait(4)
            except subprocess.TimeoutExpired:
                guard.terminate()
                guard.wait(3)
        print(json.dumps({'monitor': 'lifecycle_terminal', 'session': plan['session'],
            'state': state['phase'], 'cleanup_verified': state['cleanup_verified'],
            'evidence_errors': state['evidence_errors']}, allow_nan=False), flush=True)
    if primary is not None:
        raise primary
    if cleanup_error is not None:
        raise RuntimeError('Colab cleanup unresolved; inspect monitor-state.json') from cleanup_error
    require_evidence()
    return state


def watchdog(state_path, deadline, session):
    """Use the launch-time session even when state JSON is absent or damaged."""
    if type(session) is not str or not re.fullmatch(r'saal-[a-f0-9]{20}', session):
        raise ValueError('invalid watchdog session')
    while time.time() < deadline:
        try:
            state = json.loads(state_path.read_text())
            if (state.get('session') == session and state.get('terminal') is True
                    and state.get('cleanup_verified') is True):
                return
        except (OSError, ValueError, AttributeError):
            pass
        time.sleep(1)
    receipt = {'session': session, 'fired': True, 'absence_confirmed': False}
    try:
        stopped = subprocess.run(['colab', 'stop', '-s', session], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=60)
        receipt['stop_returncode'] = stopped.returncode
    except Exception as exc:
        receipt['stop_error'] = type(exc).__name__
    try:
        status = subprocess.run(['colab', 'status', '-s', session], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30)
        receipt['absence_confirmed'] = status.returncode == 0 and "Session '"+session+"' not found." in status.stdout
    except Exception as exc:
        receipt['status_error'] = type(exc).__name__
    print(json.dumps({'monitor': 'watchdog_result', **receipt}), flush=True)
    write_json(state_path.parent/'watchdog-result.json', receipt)


if __name__ == '__main__':
    # Isolated script launch has no package import path; bootstrap only this source tree.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    watchdog(Path(args.state), args.deadline, args.session)
