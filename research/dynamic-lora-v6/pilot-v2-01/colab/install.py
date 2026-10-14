import sys,subprocess,hashlib,zipfile,json,threading,time
from pathlib import Path
stop=threading.Event()
def heart():
 while not stop.wait(5):print(json.dumps({"monitor":"dependency_heartbeat"}),flush=True)
threading.Thread(target=heart,daemon=True).start()
try:
 if sys.version_info[:2]!=(3,13):raise ValueError("registered GPU Python lock mismatch")
 lock=Path("/content/saal_adaptive_lock.txt")
 if hashlib.sha256(lock.read_bytes()).hexdigest()!='fe75c6508354d719c74d50cba540d1d5306d5c74d3b77d7ebc23f1f1f34e04b4':raise ValueError("dependency lock substitution")
 subprocess.run(["uv","venv","--python",sys.executable,"/content/saal-adaptive-env"],check=True,timeout=30)
 subprocess.run(["uv","pip","install","--python","/content/saal-adaptive-env/bin/python","--require-hashes","--only-binary",":all:","-r",str(lock)],check=True,timeout=900)
 archive=Path("/content/saal_adaptive_worker.zip")
 if hashlib.sha256(archive.read_bytes()).hexdigest()!='d7add3b369afbf9a7c8a5b4d17acae6dfcff21df0d6b4a71c19f91b026cddbbf':raise ValueError("worker archive substitution")
 with zipfile.ZipFile(archive) as z:z.extractall("/content/saal-adaptive")
 print(json.dumps({"monitor":"dependency_complete","hash_lock_enforced":True}),flush=True)
except BaseException as exc:
 print(json.dumps({"monitor":"worker_failure","stage":"dependencies","detail":str(exc)}),flush=True)
 raise
finally:stop.set()
