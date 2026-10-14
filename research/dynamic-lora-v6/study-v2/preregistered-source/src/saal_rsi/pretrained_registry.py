"""Verified pinned snapshots outside release archives. No executable remote code."""
from pathlib import Path
import hashlib,json,os,urllib.request
from .pretrained_contract import entry
from .spec import canonical

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate_snapshot(root,repository,revision):
    root=Path(root);e=entry(repository,revision)
    if root.is_symlink() or not root.is_dir():raise ValueError('snapshot root')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(e['files']):raise ValueError('snapshot file inventory')
    for name,record in e['files'].items():
        p=root/name
        if p.is_symlink() or p.stat().st_size!=record['size'] or file_hash(p)!=record['sha256']:raise ValueError('snapshot bytes: '+name)
    from safetensors import safe_open
    inventory={}
    for name in e['files']:
        if name.endswith('.safetensors'):
            with safe_open(str(root/name),framework='pt',device='cpu') as f:
                for key in f.keys():
                    if key in inventory:raise ValueError('duplicate base tensor')
                    t=f.get_slice(key);inventory[key]={'shape':list(t.get_shape()),'dtype':t.get_dtype()}
    if not inventory or len(inventory)>3000:raise ValueError('base tensor inventory')
    return {'repository':repository,'revision':revision,'files_digest':e['model_files_digest'],'tensor_inventory':inventory}

def fetch_snapshot(cache,repository,revision,progress=lambda value:None):
    e=entry(repository,revision);root=Path(cache)/repository.split('/')[-1]/revision;root.mkdir(parents=True,exist_ok=True)
    for name,record in e['files'].items():
        p=root/name
        if p.exists():
            if p.is_symlink() or p.stat().st_size!=record['size'] or file_hash(p)!=record['sha256']:raise ValueError('existing cache substitution')
            continue
        temporary=root/(name+'.partial');progress({'monitor':'snapshot_download','file':name,'bytes':record['size']})
        with urllib.request.urlopen(f'https://huggingface.co/{repository}/resolve/{revision}/{name}',timeout=60) as response,temporary.open('xb') as f:
            count=0
            while chunk:=response.read(4*1024*1024):
                count+=len(chunk)
                if count>record['size']:raise ValueError('snapshot exceeds declared size')
                f.write(chunk)
        if temporary.stat().st_size!=record['size'] or file_hash(temporary)!=record['sha256']:raise ValueError('download hash mismatch')
        os.replace(temporary,p)
    validate_snapshot(root,repository,revision);return root
