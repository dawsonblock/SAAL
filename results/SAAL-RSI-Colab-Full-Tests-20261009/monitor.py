from pathlib import Path
import subprocess,json
p=Path('/content/saal-tests-r3/status.json');print(p.read_text() if p.exists() else 'NOT_STARTED')
if p.exists():
 s=json.loads(p.read_text());log=Path('/content/saal-tests-r3',s['stage']+'.log')
 if log.exists():print(log.read_text()[-1800:])
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
