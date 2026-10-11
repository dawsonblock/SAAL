"""Streaming Colab lifecycle monitoring and a separate named-session watchdog.

No authority keys or qualification data are used. Same-UID custody is trusted.
"""
import datetime
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def stream_command(command,log_path,timeout,on_event=lambda value:None):
    if type(timeout)not in (int,float) or not math.isfinite(timeout) or timeout<=0:raise ValueError('command deadline')
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
    started=time.monotonic();tail=bytearray();pending=bytearray();total=0
    try:
        with Path(log_path).open('xb') as log:
            while selector.get_map():
                if time.monotonic()-started>=timeout:raise TimeoutError('Colab command deadline: '+command[1])
                for key,_ in selector.select(min(.5,timeout)):
                    chunk=os.read(key.fileobj.fileno(),65536)
                    if not chunk:selector.unregister(key.fileobj);continue
                    total+=len(chunk)
                    if total>10_000_000:raise ValueError('command log budget exceeded')
                    log.write(chunk);log.flush();tail.extend(chunk);pending.extend(chunk)
                    if len(tail)>100000:del tail[:-100000]
                    if len(pending)>100000:raise ValueError('unterminated command output frame')
                    while b'\n' in pending:
                        line,_,rest=pending.partition(b'\n');pending=bytearray(rest)
                        text=line.decode(errors='replace');begin=text.find('{"monitor"')
                        if begin>=0:
                            try:record=json.loads(text[begin:])
                            except ValueError:continue
                            on_event(record)
            rc=proc.wait(timeout=max(.01,timeout-(time.monotonic()-started)))
        if rc:raise RuntimeError(f'Colab {command[1]} failed ({rc}): '+tail.decode(errors='replace')[-1200:])
        return tail.decode(errors='replace')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(2)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(2)
        selector.close();proc.stdout.close()


def launch_watchdog(state_path,deadline,session):
    log=state_path.parent/'watchdog.log'
    with log.open('xb') as output:
        return subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),'--watchdog','--state',str(state_path),
             '--deadline',str(deadline),'--session',session],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)


def execute_colab_plan(plan,workspace):
    workspace=Path(workspace).resolve();state_path=workspace/'monitor-state.json';lock=threading.RLock();done=threading.Event()
    started=time.monotonic();deadline=time.time()+plan['estimated_max_seconds']
    state={'schema':'saal-colab-monitor-v2','session':plan['session'],'phase':'PREPARED','job_digest':plan['job_digest'],
       'deadline_epoch':deadline,'heartbeat_epoch':time.time(),'terminal':False,'cleanup_verified':False,
       'progress_step':0,'total_steps':0,'external_deployment_allowed':False}
    def publish():
        with lock:
            state['heartbeat_epoch']=time.time();state['elapsed_seconds']=time.monotonic()-started;write_json(state_path,state)
    def event(kind,**value):
        with lock:
            record={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':kind,**value}
            with (workspace/'events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if kind=='remote_monitor':
                state['last_remote_event']=value
                if value.get('monitor')=='campaign_progress':
                    state['completed_runs']=value['completed_runs'];state['total_runs']=value['total_runs'];state['current_seed']=value['seed'];state['current_mode']=value['mode']
                if value.get('monitor')=='training_progress':
                    state['progress_step']=value['completed_steps'];state['total_steps']=value['total_steps'];state['latest_objective_loss']=value['objective_loss']
        publish()
    def heartbeat():
        while not done.wait(5):publish()
    publish();watchdog=launch_watchdog(state_path,deadline+10,plan['session']);state['watchdog_pid']=watchdog.pid
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start();attempted=False;primary=None
    phases=plan['phases']
    if len(phases)!=len(plan['commands'])-1:raise ValueError('phase/command inventory')
    try:
        for number,(command,phase) in enumerate(zip(plan['commands'][:-1],phases)):
            attempted=True;state['phase']=phase;state['active_command']=command[1];event('command_started',phase=phase)
            stream_command(command,workspace/(str(number)+'-'+phase.lower()+'.log'),deadline-time.time(),lambda r:event('remote_monitor',**r))
            event('command_finished',phase=phase)
    except BaseException as exc:
        primary=exc;state['error']={'type':type(exc).__name__,'detail':str(exc)};event('run_failed',error=state['error'])
        raise
    finally:
        cleanup_error=None
        if attempted:
            try:
                state['phase']='CLEANUP';event('cleanup_started')
                stream_command(plan['commands'][-1],workspace/'cleanup-stop.log',60)
                confirmation=stream_command(['colab','status','-s',plan['session']],workspace/'cleanup-status.log',30)
                if "Session '"+plan['session']+"' not found." not in confirmation:raise RuntimeError('Runtime absence not confirmed')
                state['cleanup_verified']=True;event('cleanup_confirmed')
            except BaseException as exc:
                cleanup_error=exc;state['cleanup_error']={'type':type(exc).__name__,'detail':str(exc)};event('cleanup_failed',error=state['cleanup_error'])
        state['phase']='COMPLETED' if primary is None and cleanup_error is None else 'FAILED'
        state['active_command']=None;state['terminal']=True;publish();done.set();thread.join(timeout=1)
        if state['cleanup_verified']:
            try:watchdog.wait(4)
            except subprocess.TimeoutExpired:watchdog.terminate();watchdog.wait(3)
        if primary is None and cleanup_error is not None:raise RuntimeError('Colab cleanup failed; inspect monitor-state.json') from cleanup_error
    return state


def watchdog(state_path,deadline,session):
    if not re.fullmatch(r'saal-[a-f0-9]{20}',session):raise ValueError('invalid watchdog session')
    while time.time()<deadline:
        try:
            state=json.loads(state_path.read_text())
            if state.get('terminal') and state.get('cleanup_verified'):return
        except (OSError,ValueError):pass
        time.sleep(1)
    state=json.loads(state_path.read_text())
    if state.get('session')!=session:raise ValueError('watchdog session binding changed')
    try:
        result=subprocess.run(['colab','stop','-s',session],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
        status=subprocess.run(['colab','status','-s',session],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30)
        receipt={'session':session,'fired':True,'stop_returncode':result.returncode,
                 'absence_confirmed':status.returncode==0 and "Session '"+session+"' not found." in status.stdout}
    except Exception as exc:receipt={'session':session,'fired':True,'error':type(exc).__name__}
    write_json(state_path.parent/'watchdog-result.json',receipt)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watchdog',action='store_true');p.add_argument('--state',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--session',required=True)
    args=p.parse_args()
    if not args.watchdog:raise ValueError('watchdog mode required')
    watchdog(Path(args.state),args.deadline,args.session)
