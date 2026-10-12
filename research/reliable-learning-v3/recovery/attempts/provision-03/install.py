import sys,subprocess,hashlib,zipfile,json,threading,time
from pathlib import Path
stop=threading.Event()
def heart():
 while not stop.wait(5):print(json.dumps({'monitor':'dependency_heartbeat','at':time.time()}),flush=True)
threading.Thread(target=heart,daemon=True).start()
try:
 if sys.version_info[:2]!=(3,13):raise ValueError('GPU Python lock mismatch')
 lock=Path('/content/saal_pretrained_lock.txt')
 if hashlib.sha256(lock.read_bytes()).hexdigest()!='fe75c6508354d719c74d50cba540d1d5306d5c74d3b77d7ebc23f1f1f34e04b4':raise ValueError('dependency hash mismatch')
 subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-pretrained-env'],check=True,timeout=30)
 subprocess.run(['uv','pip','install','--python','/content/saal-pretrained-env/bin/python','--require-hashes','-r',str(lock)],check=True,timeout=900)
 with zipfile.ZipFile('/content/saal_pretrained_worker.zip') as z:z.extractall('/content/saal-pretrained')
finally:stop.set()
print(json.dumps({'monitor':'dependency_complete','hash_lock_enforced':True}),flush=True)
