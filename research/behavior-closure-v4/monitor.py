"""Read-only, per-run progress derived from preserved events, with no authority."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'research/reliable-learning-v3/tools'))
from monitor_progress import progress


def snapshot():
    stages={};completed=0
    for name,folder in [('original-paired',PROJECT/'experiment/colab-paired'),
                        ('continuation',PROJECT/'experiment/continuation-01/colab'),
                        ('full',PROJECT/'experiment/colab-full')]:
        state_file=folder/'monitor-state.json'
        if not state_file.exists():continue
        state=json.loads(state_file.read_text());log=folder/'4-campaign.log'
        item=progress(log.read_text().splitlines()) if log.exists() else progress([])
        item.update(session=state['session'],controller_phase=state['phase'],controller_heartbeat_age_seconds=max(0,time.time()-state['heartbeat_epoch']),
                    terminal=state['terminal'],cleanup_verified=state['cleanup_verified'])
        reconciled=folder/'cleanup-reconciliation-02.json'
        if reconciled.exists():
            record=json.loads(reconciled.read_text());item.update(stage='INTERRUPTED',terminal=True,cleanup_verified=record['cleanup_verified'],reconciled_incident=True)
        outcomes=folder/'outcomes.json'
        item['recovered_completed_runs']=sum(r['state']=='COMPLETED' for r in json.loads(outcomes.read_text())['runs']) if outcomes.exists() else 0
        completed+=item['recovered_completed_runs'];stages[name]=item
    return {'stages':stages,'completed_new_model_runs':completed,'registered_new_model_runs':11,
            'controller_cpu_replay_records':len(list((PROJECT/'cpu-replay').glob('*.json'))),
            'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
            'scope':'Read-only public progress. New run boundaries clear stale counters. A controller heartbeat is not independent GPU attestation or qualification.'}


if __name__=='__main__':print(json.dumps(snapshot(),indent=2))
