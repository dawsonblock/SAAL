"""Read-only derived progress: reset stale step counters at each run boundary."""
from pathlib import Path
import json,time,sys
ROOT=Path(__file__).resolve().parents[1]
def progress(lines):
 current={'stage':'PREPARING','step':None,'loss':None,'tokens':None}
 for line in lines:
  try:event=json.loads(line)
  except ValueError:continue
  kind=event.get('monitor')
  if kind=='campaign_progress':
   current={'stage':'COMPLETED' if event['mode']=='TERMINAL' else 'INITIAL_MEASUREMENTS','completed_runs':event['completed_runs'],'total_runs':event['total_runs'],'seed':event['seed'],'mode':event['mode'],'step':None,'loss':None,'tokens':None}
  elif kind=='training_progress':current.update(stage='TRAINING',step=event['completed_steps'],total_steps=event['total_steps'],loss=event['objective_loss'],tokens=event['tokens'],gpu_allocated_bytes=event['gpu_allocated_bytes'])
  elif kind=='checkpoint_metrics':current.update(stage='PUBLIC_CHECKPOINT_MEASUREMENT',checkpoint_step=event['step'],development_accuracy=event['development']['accuracy'],critical_development_proposals=event['development']['unauthorized_proposals'])
  elif kind=='training_complete':current.update(stage='EXPORT_AND_REPLAY')
  elif kind=='fresh_process_replay':current.update(stage='ARTIFACT_RECOVERY',fresh_process_replay=event['verified'])
  elif kind=='worker_failure':current.update(stage='FAILED',error=event)
 return current
def snapshot():
 cloud=ROOT/'experiment/colab'
 if (ROOT/'recovery/experiment/colab/monitor-state.json').exists():cloud=ROOT/'recovery/experiment/colab'
 pointer=ROOT/'recovery/experiment/signed_execution_location.json'
 if pointer.exists():cloud=ROOT/json.loads(pointer.read_text())['payload']['relative_workspace']
 log=cloud/'4-campaign.log';p=progress(log.read_text().splitlines()) if log.exists() else progress([])
 state=json.loads((cloud/'monitor-state.json').read_text());p.update(session=state['session'],lifecycle_phase=state['phase'],heartbeat_epoch=state['heartbeat_epoch'],cleanup_verified=state['cleanup_verified'],qualification_status='NOT_EVALUATED',external_deployment_allowed=False,scope='Derived public monitoring, no authority; lifecycle completion alone does not mean tests passed.')
 if state.get('error'):p.update(stage='FAILED',lifecycle_error=state['error'])
 qa={}
 for name in ('colab-tests','colab-tests-02','colab-tests-03'):
  folder=ROOT/name;status=folder/'monitor-state.json'
  if not status.exists():continue
  q=json.loads(status.read_text());item={'phase':q['phase'],'cleanup_verified':q['cleanup_verified']};result=folder/'evidence/result.json'
  if result.exists():
   codes=json.loads(result.read_text())['returncodes'];item.update(returncodes=codes,test_outcome='PASS' if all(v==0 for v in codes.values()) else 'FAIL')
  qa[name]=item
 p['qa_sessions']=qa;p['original_completed_runs']=4 if 'recovery' in cloud.parts else 0;p['completed_primary_roles']=p.get('completed_runs',0)+p['original_completed_runs'];(ROOT/'live-progress.json').write_text(json.dumps(p,indent=2)+'\n');return p,state['terminal']
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='watch':
  deadline=time.monotonic()+44000
  while time.monotonic()<deadline:
   p,terminal=snapshot()
   if terminal:break
   time.sleep(10)
 else:print(json.dumps(snapshot()[0],indent=2))
