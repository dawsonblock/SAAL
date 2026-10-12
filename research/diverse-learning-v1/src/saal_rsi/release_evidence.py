"""Controller-owned source/evidence integrity, never a qualification authority."""
from pathlib import Path
import hashlib
from .model_contract import verify
from .spec import parse_unique_json,sha

def verify_release_evidence(root,pin):
    root=Path(root).resolve();envelope=parse_unique_json((root/'release-evidence/signed-statement.json').read_bytes());p=verify(envelope,pin)
    fields={'schema','version','signer_role','source_inventory_digest','evidence_files','qualification_status','external_deployment_allowed','limits'}
    if set(p)!=fields or p['schema']!='saal-controller-release-evidence-v1' or p['version']!='0.9.4' or p['signer_role']!='CONTROLLER_RESEARCH_ATTRIBUTION' or p['qualification_status']!='NOT_EVALUATED' or p['external_deployment_allowed']is not False:raise ValueError('release evidence scope')
    inventory=parse_unique_json((root/'release-evidence/source-inventory.json').read_bytes())
    if sha(inventory)!=p['source_inventory_digest'] or set(inventory)!= {'schema','files'} or inventory['schema']!='saal-source-inventory-v1':raise ValueError('source inventory identity')
    def check(files):
        if type(files)is not dict or not files:raise ValueError('evidence inventory')
        for name,digest in files.items():
            if type(name)is not str:raise ValueError('evidence path')
            q=root/name
            if q.is_symlink() or not q.resolve().is_relative_to(root) or not q.is_file():raise ValueError('evidence path boundary')
            h=hashlib.sha256()
            with q.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=digest:raise ValueError('release source/evidence substitution')
    check(inventory['files']);check(p['evidence_files'])
    return {'source_files_verified':len(inventory['files']),'evidence_files_verified':len(p['evidence_files']),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'limits':p['limits']}
