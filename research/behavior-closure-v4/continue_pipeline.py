"""Bounded in-turn sequencing; never restart a consumed training reservation."""
from pathlib import Path
import json, sys, time, subprocess, hashlib
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import canonical


def completed_qa():
    state_file=PROJECT/'colab-tests/monitor-state.json'
    if not state_file.exists():return False
    state=json.loads(state_file.read_text())
    if not state['terminal']:return False
    if not state['cleanup_verified']:raise RuntimeError('QA cleanup unresolved')
    result=json.loads((PROJECT/'colab-tests/evidence/result.json').read_text())
    if result['returncodes']!={'core':0,'corpus':0}:raise RuntimeError('current Colab QA failed')
    for name,n,skip in [('core',635,1),('corpus',38,0)]:
        suites=ET.parse(PROJECT/f'colab-tests/evidence/{name}.xml').getroot().iter('testsuite')
        counts={k:0 for k in ('tests','failures','errors','skipped')}
        for suite in suites:
            for k in counts:counts[k]+=int(suite.get(k,0))
        if counts!={'tests':n,'failures':0,'errors':0,'skipped':skip}:raise RuntimeError('QA count or failures differ from admitted source')
    return True


def head_absent():
    p=PROJECT/'legacy-head/colab/monitor-state.json'
    if not p.exists():return False
    s=json.loads(p.read_text())
    if not s['terminal']:return False
    if not s['cleanup_verified']:raise RuntimeError('head-control cleanup unresolved')
    return True


def run():
    deadline=time.monotonic()+9000
    while not (completed_qa() and head_absent()):
        if time.monotonic()>deadline:raise TimeoutError('bounded prerequisite wait expired')
        print(json.dumps({'monitor':'pipeline_prerequisite_wait','qa_complete':completed_qa(),'head_runtime_absent':head_absent()}),flush=True)
        time.sleep(30)
    for stage in ('paired','full'):
        print(json.dumps({'monitor':'pipeline_stage_start','stage':stage}),flush=True)
        subprocess.run([sys.executable,'-u',str(PROJECT/'campaign.py'),'run',stage],cwd=ROOT,check=True,
            timeout=39300 if stage=='paired' else 5100)
    print(json.dumps({'monitor':'pipeline_complete','qualification_status':'NOT_EVALUATED'}),flush=True)


if __name__=='__main__':run()
