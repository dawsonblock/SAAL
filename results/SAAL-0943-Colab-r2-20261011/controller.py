from pathlib import Path
import subprocess,time,json
r=Path(__file__).resolve().parent;s='saal-0943-audit-r2-20261011';events=[]
def call(args,timeout=120):
 p=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout);events.append({'args':args,'time':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'controller-events.json').write_text(json.dumps(events,indent=2));print(' '.join(args),p.stdout,p.stderr,flush=True);return p
try:
 parts=json.loads((r/'parts.json').read_text())
 for name in [*(p['name'] for p in parts['parts']),'parts.json']:
  assert call(['upload','--session',s,str(r/name),'/content/'+name]).returncode==0
 assert call(['exec','--session',s,'--file',str(r/'remote.py'),'--timeout','2100'],2160).returncode==0
 assert call(['download','--session',s,'/content/saal-0943-evidence.zip',str(r/'remote-evidence.zip')]).returncode==0
finally:
 stop=call(['stop','--session',s]);(r/'stop.json').write_text(json.dumps({'returncode':stop.returncode,'stdout':stop.stdout,'stderr':stop.stderr},indent=2))
 listing=call(['sessions']);(r/'sessions-after.json').write_text(json.dumps({'returncode':listing.returncode,'stdout':listing.stdout,'stderr':listing.stderr},indent=2))
