"""Resume only missing reconstruction records, never model-training attempts."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/reliable-learning-v3/tools'))
from study_layout import load_runs


if __name__=='__main__':
    prior=ROOT/'research/reliable-learning-v3';pin=(prior/'experiment/controller.pub').read_text().strip()
    runs,_=load_runs(prior,pin);output=PROJECT/'v3-cpu-replay'
    for run in runs:
        if run['role_index']<4:continue
        target=output/(run['run']+'.json')
        if target.exists():
            record=json.loads(target.read_text())
            if record['job_digest']!=run['job_digest']:raise ValueError('existing replay substitution')
            continue
        subprocess.run([sys.executable,'-u',str(prior/'tools/reproduce_cpu.py'),str(run['role_index']),sys.argv[1],
                        '--output',str(output)],cwd=ROOT,check=True,timeout=2000)
