import subprocess,json,sys
raw=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],text=True,timeout=15)
rows=[r.strip().split(",") for r in raw.strip().splitlines()]
if len(rows)!=1 or "T4" not in rows[0][0] or int(rows[0][2].strip())<6144:raise ValueError("controller T4/memory observation failed")
print(json.dumps({"monitor":"controller_gpu_observation","nvidia_smi":raw,"python":sys.version,"scope":"provider runtime command outside training child; not hostile-host attestation"}),flush=True)
