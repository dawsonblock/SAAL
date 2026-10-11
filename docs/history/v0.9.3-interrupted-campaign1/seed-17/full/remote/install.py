import hashlib,json,shutil,subprocess,sys,threading,time
from pathlib import Path
done=threading.Event()
def heartbeat():
    while not done.wait(5):print(json.dumps({'monitor':'dependency_heartbeat','at':time.time()}),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
try:
    if sys.version_info[:2]!=(3,13):raise ValueError('GPU lock requires Python 3.13')
    lock=Path('/content/saal_gpu_lock.txt')
    if hashlib.sha256(lock.read_bytes()).hexdigest()!='d3c8635a4d34a6a6fda8a0c1fbb2a1141eba8838d314080cc45be27d885c0ced':raise ValueError('lock digest')
    if not shutil.which('uv'):raise ValueError('reviewed uv installer unavailable')
    subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-env'],check=True,timeout=30)
    subprocess.run(['uv','pip','install','--python','/content/saal-env/bin/python','--require-hashes','-r',str(lock)],check=True,timeout=900)
finally:done.set()
print(json.dumps({'monitor':'dependency_complete','hash_lock_enforced':True}),flush=True)
