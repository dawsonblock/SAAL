import subprocess,os,json,zipfile,hashlib,traceback
from pathlib import Path
try:
 env=dict(os.environ,SAAL_HASH_LOCK_ENFORCED="1")
 commands=[['/content/saal-adaptive-env/bin/python', '-I', '/content/saal-adaptive/worker.py', '--job', '/content/saal_adaptive_job.json', '--public-key', 'e3db435cf14ebd609a7950c039a15809001bf00b23fa7deb363a7f1bffe0df4b', '--out', '/content/run-00-75e7a094f934', '--cache', '/content/saal-model-cache'], ['/content/saal-adaptive-env/bin/python', '-I', '/content/saal-adaptive/worker.py', '--job', '/content/saal_adaptive_job.json', '--public-key', 'e3db435cf14ebd609a7950c039a15809001bf00b23fa7deb363a7f1bffe0df4b', '--out', '/content/run-00-75e7a094f934', '--cache', '/content/saal-model-cache', '--replay']]
 for command in commands:
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
  for line in iter(p.stdout.readline,b""):print(line.decode(errors="replace"),end="",flush=True)
  if p.wait():raise RuntimeError("adaptive training or fresh replay child failed")
 with zipfile.ZipFile('/content/run-00-75e7a094f934.zip',"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n in ['adapter_config.json', 'artifact_inventory.json', 'best.safetensors', 'best_rank_pattern.json', 'checkpoint_selection.json', 'dataset_manifest.json', 'fresh_process_replay.json', 'initial.safetensors', 'initial_rank_pattern.json', 'model_spec.json', 'rank_history.jsonl', 'rank_schedule.json', 'resource_accounting.json', 'result.json', 'sampler_manifest.json', 'signed_training_job.json', 'snapshot_manifest.json', 'terminal.safetensors', 'terminal_rank_pattern.json', 'training_metrics.jsonl']:z.write(Path('/content/run-00-75e7a094f934')/n,n)
 transfer=Path('/content/run-00-75e7a094f934-transfer');transfer.mkdir()
 chunks=[];total=0;whole=hashlib.sha256()
 with Path('/content/run-00-75e7a094f934.zip').open("rb") as source:
  i=0
  while b:=source.read(32000000):
   n=f"part-{i:03d}.bin";(transfer/n).write_bytes(b);chunks.append({"name":n,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()});whole.update(b);total+=len(b);i+=1
 print(json.dumps({"monitor":"artifact_transfer_ready","manifest":{"schema":"saal-artifact-transfer-v1","job_digest":'sha256:75e7a094f93460c45243936d9127987859a1870c9aa73f4b7094bafb6e270c07',"size":total,"sha256":whole.hexdigest(),"chunks":chunks}}),flush=True)
except BaseException as exc:
 print(json.dumps({"monitor":"worker_failure","stage":"adaptive_train_replay_export","error_type":type(exc).__name__,"detail":str(exc)}),flush=True)
 traceback.print_exc()
 raise
