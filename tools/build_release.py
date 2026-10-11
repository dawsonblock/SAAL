"""Freeze a complete source manifest and ZIP. Run qualification after final source edits."""
from pathlib import Path
import hashlib
import sys
import zipfile

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve()
if out.exists():raise FileExistsError('refuse to overwrite output')
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and '.pytest_cache' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts) and p.suffix not in {'.pyc','.pyo'}
             and p.name!='MANIFEST.sha256')
if any(p.is_symlink() for p in files):raise ValueError('symlinks forbidden')
(root/'MANIFEST.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix()+'\n' for p in files))
files=sorted(files+[root/'MANIFEST.sha256'])
out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for p in files:
        info=zipfile.ZipInfo(root.name+'/'+p.relative_to(root).as_posix(),date_time=(2026,10,9,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        archive.writestr(info,p.read_bytes())
print('Archive:',out,'files:',len(files),'bytes:',out.stat().st_size)
print('SHA256:',hashlib.sha256(out.read_bytes()).hexdigest())
