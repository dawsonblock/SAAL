"""Controller attribution of corrective source and evidence, no qualification."""
from pathlib import Path
import hashlib
from .model_contract import verify
from .spec import parse_unique_json,sha


def verify_capsule(root,pin):
    root=Path(root).resolve();directory=root/'corrective-evidence'
    body=verify(parse_unique_json((directory/'signed-statement.json').read_bytes()),pin)
    if type(body)is not dict or set(body)!={'schema','version','role','source_inventory_digest','historical_freeze_digest','evidence_files','qualification_status','independent_deployment_verified','external_deployment_allowed','limits'} or body['schema']!='saal-corrective-controller-evidence-v1' or body['version']!='0.9.4.1' or body['role']!='CONTROLLER_RESEARCH_ATTRIBUTION' or body['qualification_status']!='NOT_EVALUATED' or body['independent_deployment_verified']is not False or body['external_deployment_allowed']is not False:
        raise ValueError('corrective capsule cannot claim independent authority')
    inventory=parse_unique_json((directory/'source-inventory.json').read_bytes())
    freeze=parse_unique_json((root/'provenance/v0.9.4.1-historical-input.json').read_bytes())
    if sha(inventory)!=body['source_inventory_digest'] or sha(freeze)!=body['historical_freeze_digest'] or set(inventory)!={'schema','files'} or inventory['schema']!='saal-source-inventory-v1':raise ValueError('capsule source/history identity')
    for files in (inventory['files'],body['evidence_files']):
        if type(files)is not dict or not files:raise ValueError('capsule inventory')
        for name,digest in files.items():
            if type(name)is not str or not name:raise ValueError('capsule path')
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():raise ValueError('capsule path boundary')
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=digest:raise ValueError('corrective source/evidence substitution')
    return {'version':'0.9.4.1','source_files_verified':len(inventory['files']),'evidence_files_verified':len(body['evidence_files']),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
