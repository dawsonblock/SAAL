"""Controller reproduction as each frozen GPU bundle arrives."""
import json,sys,time
from pathlib import Path
from replay_results import replay,ROOT
from verify_capsule import verify_recorded_replay

if __name__=='__main__':
    deadline=time.monotonic()+15000
    screens=[]
    for index in range(3):
        while True:
            if time.monotonic()>deadline:raise TimeoutError('controller wait budget')
            p=ROOT/'experiment/colab/outcomes.json'
            outcomes=json.loads(p.read_text())['runs'] if p.exists() else []
            if len(outcomes)>index:break
            time.sleep(10)
        if outcomes[index]['state']!='COMPLETED':raise RuntimeError('preserved interrupted GPU attempt')
        run=outcomes[index];destination=ROOT/'experiment/controller-replay'/(run['run']+'.json')
        if not destination.exists():
            try:replay(index,sys.argv[1])
            except ValueError:
                if not destination.exists():raise
                # replay writes its complete failing numerical record before raising.
        record=json.loads(destination.read_text());result=json.loads((ROOT/'experiment/artifacts'/run['run']/'result.json').read_text())
        measured=verify_recorded_replay(record,result,run['job_digest']);screens.append(measured|{'run':run['run']})
        print(json.dumps({'controller_cpu_replay_record':run['run'],**measured}),flush=True)
    summary={'runs':screens,'all_recorded':True,'all_probability_screens_pass':all(s['passes_registered_probability_tolerance'] for s in screens),'all_decisions_match':all(s['all_decisions_match'] for s in screens),'qualification_status':'NOT_EVALUATED'}
    with (ROOT/'experiment/controller-replay-status.json').open('x') as f:json.dump(summary,f,indent=2)
