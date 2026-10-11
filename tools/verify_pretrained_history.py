"""Account for every inherited byte change; preserve the original release ZIP."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1];freeze=json.loads((root/'provenance/v0.9.4-historical-input.json').read_text());changed=[];missing=[]
for name,digest in freeze['files'].items():
    p=root/name
    if not p.is_file():missing.append(name)
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
allowed={'README.md','RELEASE_REPORT.md','pyproject.toml','src/saal_rsi/__init__.py','tools/build_release.py'}
if missing or set(changed)-allowed:raise ValueError({'unexplained_changes':sorted(set(changed)-allowed),'missing':missing})
archive=root/'provenance/SAAL-RSI-v0.9.3-Full-Corrected-Upgraded.zip'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=freeze['sha256']:raise ValueError('historical input ZIP changed')
print(json.dumps({'historical_files_verified':len(freeze['files'])-len(changed),'accounted_changes':changed,'original_zip_unchanged':True}))
