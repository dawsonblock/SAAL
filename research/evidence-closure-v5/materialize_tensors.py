"""Restore exact logical checkpoint paths from complete content-addressed bytes."""
from pathlib import Path
import hashlib,json,os,re


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def materialize(root):
    root=Path(root).resolve();mapping=json.loads((root/'TENSOR_PAYLOADS.json').read_text())
    if set(mapping)!={'schema','files'} or mapping['schema']!='saal-physical-tensor-payloads-v1':raise ValueError('tensor manifest schema')
    for name,entry in mapping['files'].items():
        if type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts or not name.endswith('.safetensors'):
            raise ValueError('checkpoint path escape')
        if type(entry)is not dict or set(entry)!={'sha256','size','payload'} or not re.fullmatch('[0-9a-f]{64}',entry['sha256']) or type(entry['size'])is not int or not 1<=entry['size']<=650_000_000 or entry['payload']!='tensor_payloads/'+entry['sha256']+'.safetensors':
            raise ValueError('checkpoint payload identity')
        source=root/entry['payload'];destination=root/name
        for parent in (source.parent,destination.parent):
            if not parent.resolve().is_relative_to(root):raise ValueError('symlink parent escape')
        if source.is_symlink() or not source.is_file() or source.stat().st_size!=entry['size'] or digest(source)!=entry['sha256']:
            raise ValueError('missing/substituted physical checkpoint bytes')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file() or destination.stat().st_size!=entry['size'] or digest(destination)!=entry['sha256']:
                raise FileExistsError('existing checkpoint differs; never overwrite')
        else:os.link(source,destination)
    return len(mapping['files'])


if __name__=='__main__':
    import sys
    print('Restored exact checkpoint paths:',materialize(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent))
