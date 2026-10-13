"""Clean extraction with optional verified tensor hardlinks to save local disk.

Every ZIP member is read completely. Reuse saves a second physical copy of an
identical tensor; it never replaces authentication of the archive's own bytes.
"""
from pathlib import Path
import hashlib,os,re,stat,zipfile


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def extract(archive,destination,expected_sha256,reuse_roots=()):
    archive=Path(archive);destination=Path(destination)
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or digest(archive)!=expected_sha256:
        raise ValueError('archive authentication failed')
    if destination.exists() or destination.is_symlink():raise FileExistsError('clean destination required')
    reusable={}
    for root in reuse_roots:
        for p in Path(root).rglob('*.safetensors'):
            if p.is_file() and not p.is_symlink() and p.stat().st_size>50_000_000:
                reusable[digest(p)]=p
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(infos)>10000 or len(names)!=len(set(n.casefold() for n in names)) or sum(i.file_size for i in infos)>20_000_000_000:
            raise ValueError('ZIP inventory bounds or duplicate paths')
        for info in infos:
            name=info.filename;p=Path(name)
            if not name or p.is_absolute() or '..' in p.parts or '\\' in name or info.is_dir() or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('unsafe ZIP member')
        destination.mkdir(parents=True);linked=[]
        for info in infos:
            target=destination/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
            match=re.fullmatch('tensor_payloads/([0-9a-f]{64})\\.safetensors',info.filename)
            candidate=reusable.get(match[1]) if match else None
            if candidate is not None and candidate.stat().st_size==info.file_size:
                h=hashlib.sha256()
                with z.open(info) as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=match[1] or digest(candidate)!=match[1]:raise ValueError('tensor reuse byte mismatch')
                os.link(candidate,target);linked.append(info.filename)
            else:
                with z.open(info) as source,target.open('xb') as output:
                    for chunk in iter(lambda:source.read(1024*1024),b''):output.write(chunk)
    return {'clean_directory':str(destination),'archive_sha256':expected_sha256,'members_read_completely':len(infos),
            'verified_tensor_hardlinks':linked,'linked_source_bytes_are_shared':True}


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('destination');p.add_argument('sha256');p.add_argument('--reuse-root',action='append',default=[])
    a=p.parse_args();print(json.dumps(extract(a.archive,a.destination,a.sha256,a.reuse_root),indent=2))
