"""Verify v0.9.4 preservation; allow only enumerated release metadata drift."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
freeze=json.loads((root/'provenance/v0.9.4.1-historical-input.json').read_text())
allowed={'README.md','RELEASE_REPORT.md','pyproject.toml','src/saal_rsi/__init__.py','MANIFEST.sha256'}
changed=[]
for name,digest in freeze['files'].items():
    path=root/name
    if path.is_symlink() or not path.is_file():raise ValueError('historical file missing or substituted: '+name)
    with path.open('rb') as stream:current=hashlib.file_digest(stream,'sha256').hexdigest()
    if current!=digest:changed.append(name)
if set(changed)-allowed:raise ValueError('unexplained historical source/evidence drift: '+str(sorted(set(changed)-allowed)))
archive=root/'provenance'/freeze['archive']
with archive.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=freeze['archive_sha256']:raise ValueError('original v0.9.4 ZIP substituted')
print(json.dumps({'historical_files_verified':len(freeze['files'])-len(changed),'accounted_metadata_changes':sorted(changed),'original_archive_sha256':freeze['archive_sha256'],'historical_neural_workers_unchanged':True,'historical_checkpoints_reports_and_receipts_unchanged':True}))
