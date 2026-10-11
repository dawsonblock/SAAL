from pathlib import Path
import subprocess,json,time
r=Path(__file__).resolve().parent;session='saal-corpus-v1-20261010';events=[]
def call(args,timeout=120):
 start=time.time();p=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout);events.append({'args':args,'started':start,'duration':time.time()-start,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'controller-events.json').write_text(json.dumps(events,indent=2));print(' '.join(args),p.stdout,p.stderr,flush=True);return p
try:
 for local,remote in [(Path('/Users/dawsonblock/Downloads/SAAL-RSI-Decision-Corpus-v1-Review-Candidate.zip'),'corpus.zip'),(r/'base-source.zip','base-source.zip'),(r/'INPUTS.json','INPUTS.json')]:
  p=call(['upload','--session',session,str(local),'/content/'+remote]);assert p.returncode==0
 p=call(['exec','--session',session,'--file',str(r/'remote_run.py'),'--timeout','600'],660);assert p.returncode==0
 p=call(['download','--session',session,'/content/saal-corpus-evidence.zip',str(r/'remote-evidence.zip')]);assert p.returncode==0
finally:
 p=call(['stop','--session',session]);(r/'cleanup-stop.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
 p=call(['sessions']);(r/'cleanup-sessions.json').write_text(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2))
