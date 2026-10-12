import subprocess,os,zipfile,json,hashlib
from pathlib import Path
env=dict(os.environ,SAAL_HASH_LOCK_ENFORCED='1')
command=['/content/saal-pretrained-env/bin/python', '-I', '/content/saal-pretrained/worker.py', '--job', '/content/saal_pretrained_job.json', '--public-key', 'e3c65fff17f286c6538793961f3dc9aa3676836f2205feadb7dc385d0538111e', '--out', '/content/run-01-74d7dbd2572d', '--cache', '/content/saal-model-cache']
proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
for line in iter(proc.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)
rc=proc.wait()
if rc:
 print(json.dumps({'monitor':'worker_failure','stage':'training','returncode':rc}),flush=True)
 raise RuntimeError('training child failed')
replay=subprocess.Popen(command+['--replay'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
for line in iter(replay.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)
rc=replay.wait()
if rc:
 print(json.dumps({'monitor':'worker_failure','stage':'replay','returncode':rc}),flush=True)
 raise RuntimeError('fresh-process replay failed')
with zipfile.ZipFile('/content/run-01-74d7dbd2572d.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in ['artifact_inventory.json', 'best.safetensors', 'dataset_manifest.json', 'initial.safetensors', 'model_spec.json', 'result.json', 'signed_training_job.json', 'snapshot_manifest.json', 'terminal.safetensors', 'training_metrics.jsonl']:z.write(Path('/content/run-01-74d7dbd2572d')/n,n)
transfer=Path('/content/run-01-74d7dbd2572d-transfer');transfer.mkdir()
chunks=[];total=0;whole=hashlib.sha256()
with Path('/content/run-01-74d7dbd2572d.zip').open('rb') as source:
 i=0
 while b:=source.read(32000000):
  name=f'part-{i:03d}.bin';(transfer/name).write_bytes(b);chunks.append({'name':name,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()});whole.update(b);total+=len(b);i+=1
(transfer/'manifest.json').write_text(json.dumps({'schema':'saal-artifact-transfer-v1','job_digest':'sha256:74d7dbd2572dcbae67f7fc87c62b420486a33d04175540bb1402937ba28804d9','size':total,'sha256':whole.hexdigest(),'chunks':chunks}))
