import subprocess,sys,os
from pathlib import Path
log=open('/content/saal-runner.log','w')
p=subprocess.Popen([sys.executable,'/content/remote_rerun.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/saal-runner.pid').write_text(str(p.pid))
print({'runner_pid':p.pid,'deadline_seconds':1800})
watch='import os,sys,time,signal;time.sleep(1800);pid=int(sys.argv[1]);\ntry:os.killpg(pid,signal.SIGTERM)\nexcept ProcessLookupError:pass'
subprocess.Popen([sys.executable,'-c',watch,str(p.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
