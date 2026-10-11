from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parent;s='saal-0943-package-20261011';events=[]
def call(args,timeout=120):
 p=subprocess.run(['colab',*args],capture_output=True,text=True,timeout=timeout);events.append({'args':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(r/'controller-events.json').write_text(json.dumps(events,indent=2));print(p.stdout,p.stderr,flush=True);return p
try:
 assert call(['upload','--session',s,str(r/'input.zip'),'/content/package-input.zip']).returncode==0
 assert call(['exec','--session',s,'--file',str(r/'remote.py'),'--timeout','420'],480).returncode==0
 assert call(['download','--session',s,'/content/packaging-evidence.zip',str(r/'evidence.zip')]).returncode==0
finally:
 call(['stop','--session',s]);call(['sessions'])
