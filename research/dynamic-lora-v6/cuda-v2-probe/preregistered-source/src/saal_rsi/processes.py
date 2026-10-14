"""Bounded POSIX JSON IPC. Deadline covers writes AND reads; not an OS sandbox."""
from pathlib import Path
import math
import os
import selectors
import subprocess
import sys
import tempfile
import time
from .spec import canonical,parse_unique_json

MAX_MESSAGE=65536

class Worker:
    def __init__(self,role):
        if role not in {'actor','executor'}:raise ValueError('unknown worker role')
        if os.name!='posix':raise RuntimeError('process workers require POSIX pipe selectors')
        self.directory=tempfile.TemporaryDirectory(prefix='saal-rsi-worker-')
        env={k:os.environ[k] for k in ('PATH','SYSTEMROOT') if k in os.environ}
        try:
            self.proc=subprocess.Popen([sys.executable,'-I',str(Path(__file__).with_name('worker.py')),role],
                cwd=self.directory.name,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,bufsize=0,close_fds=True)
            os.set_blocking(self.proc.stdin.fileno(),False)
            os.set_blocking(self.proc.stdout.fileno(),False)
        except BaseException:
            if hasattr(self,'proc'):self.close()
            else:self.directory.cleanup()
            raise
        self.closed=False

    def request(self,payload,timeout=10):
        if type(timeout)not in (int,float) or not math.isfinite(timeout) or timeout<0:raise ValueError('invalid timeout')
        if self.closed:raise RuntimeError('worker already closed')
        raw=canonical(payload)+b'\n'
        if len(raw)>MAX_MESSAGE:raise ValueError('request too large')
        deadline=time.monotonic()+timeout
        response=b'';offset=0
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.proc.stdin,selectors.EVENT_WRITE)
                selector.register(self.proc.stdout,selectors.EVENT_READ)
                while b'\n' not in response:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise RuntimeError('worker timeout')
                    events=selector.select(remaining)
                    if not events:raise RuntimeError('worker timeout')
                    for key,event in events:
                        if key.fileobj is self.proc.stdin:
                            try:offset+=os.write(self.proc.stdin.fileno(),raw[offset:])
                            except BlockingIOError:continue
                            if offset==len(raw):selector.unregister(self.proc.stdin)
                        else:
                            try:chunk=os.read(self.proc.stdout.fileno(),4096)
                            except BlockingIOError:continue
                            if not chunk:raise RuntimeError('worker exited')
                            response+=chunk
                            if len(response)>MAX_MESSAGE:raise ValueError('response too large')
                if offset!=len(raw) or not response.endswith(b'\n') or response.count(b'\n')!=1:
                    raise ValueError('unexpected worker response framing')
            result=parse_unique_json(response[:-1])
            if 'error' in result:raise RuntimeError('worker refused request: '+str(result['error']))
            if result.get('pid')!=self.proc.pid:raise ValueError('worker process identity mismatch')
            return result
        except BaseException as exc:
            self.close()
            if isinstance(exc,(BrokenPipeError,OSError)):raise RuntimeError('worker communication failed') from exc
            raise

    def close(self):
        if getattr(self,'closed',False):return
        self.closed=True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=2)
        for stream in (self.proc.stdin,self.proc.stdout):
            if stream:stream.close()
        self.directory.cleanup()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
