import json,traceback,shutil
try:
 import subprocess,os,zipfile,json,hashlib
 from pathlib import Path
 env=dict(os.environ,SAAL_HASH_LOCK_ENFORCED='1')
 command=['/content/saal-pretrained-env/bin/python', '-I', '/content/saal-pretrained/worker.py', '--job', '/content/saal_pretrained_job.json', '--public-key', 'ccc792a3f49c372eaa3c66a9089e02e4350e19e63f4e677768561298caf32fc2', '--out', '/content/run-02-c553fcf68a09', '--cache', '/content/saal-model-cache']
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
 with zipfile.ZipFile('/content/run-02-c553fcf68a09.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n in ['artifact_inventory.json', 'best.safetensors', 'dataset_manifest.json', 'diagnostic_manifest.json', 'initial.safetensors', 'model_spec.json', 'result.json', 'sampler_manifest.json', 'signed_training_job.json', 'snapshot_manifest.json', 'terminal.safetensors', 'training_metrics.jsonl']:z.write(Path('/content/run-02-c553fcf68a09')/n,n)
 transfer=Path('/content/run-02-c553fcf68a09-transfer');transfer.mkdir()
 chunks=[];total=0;whole=hashlib.sha256()
 with Path('/content/run-02-c553fcf68a09.zip').open('rb') as source:
  i=0
  while b:=source.read(32000000):
   name=f'part-{i:03d}.bin';(transfer/name).write_bytes(b);chunks.append({'name':name,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()});whole.update(b);total+=len(b);i+=1
 (transfer/'manifest.json').write_text(json.dumps({'schema':'saal-artifact-transfer-v1','job_digest':'sha256:c553fcf68a09a3abb716821ecd4feb34bf587e59e98c3ec30636aed71bba2ef4','size':total,'sha256':whole.hexdigest(),'chunks':chunks}))
 
 print(json.dumps({"monitor":"artifact_transfer_ready","manifest":json.loads((transfer/"manifest.json").read_text())}),flush=True)
except BaseException as exc:
 print(json.dumps({'monitor':'worker_failure','stage':'export_or_child','error_type':type(exc).__name__,'detail':str(exc)}),flush=True)
 traceback.print_exc()
 raise
