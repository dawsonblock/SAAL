"""Controller file integrity and distinct dynamic identity; never qualification."""
from pathlib import Path
import argparse,hashlib,json
from saal_rsi.spec import parse_unique_json,sha
from saal_rsi.model_contract import verify
from saal_adalora.contract import validate,source_identity

def check_files(root,files):
    root=Path(root).resolve()
    if type(files)is not dict or not files:raise ValueError('file inventory')
    for n,h in files.items():
        if type(n)is not str or not n or Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('path boundary')
        p=root/n
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):raise ValueError('file boundary')
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=h:raise ValueError('file substitution '+n)

def check(root,pin):
    root=Path(root).resolve();body=verify(parse_unique_json((root/'SIGNED_FILES.json').read_bytes()),pin)
    if set(body)!={'schema','files','qualification_status','external_deployment_allowed'} or body['schema']!='saal-adalora-research-files-v1' or body['qualification_status']!='NOT_EVALUATED' or body['external_deployment_allowed']is not False:raise ValueError('dynamic file authority')
    check_files(root,body['files']);e=root/'experiment-r2';job=verify(parse_unique_json((e/'signed-job.json').read_bytes()),pin);validate(job)
    outcome=parse_unique_json((e/'outcome.json').read_bytes())
    if outcome['state']=='COMPLETED':
        result=parse_unique_json((e/'result.json').read_bytes());replay=parse_unique_json((e/'fresh-replay.json').read_bytes())
        if result['job_digest']!=sha(job) or result['source_identity']!=source_identity() or replay['job_digest']!=sha(job) or replay['all_snapshots_reproduced']is not True:raise ValueError('dynamic reconstruction identity')
    return {'files_verified':len(body['files']),'attempt_state':outcome['state'],'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--pin',required=True);a=p.parse_args();print(json.dumps(check(a.root,a.pin)))
