"""Verify every inherited v0.9.4.1 file; permit only exact recorded repairs."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parents[1]
freeze = json.loads((root/'provenance/v0.9.4.2-historical-input.json').read_text())
changes = json.loads((root/'corrective-evidence/validation-repair-changes.json').read_text())
actual_changes = set()
for name, before in freeze['files'].items():
    path = root/name
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing/substituted historical file: '+name)
    if name == 'corrective-evidence/signed-statement.json':
        if changes.get(name) != {'before': before, 'after': 'REGENERATED_SIGNATURE_VERIFIED_SEPARATELY'}:
            raise ValueError('signature regeneration must be declared')
        actual_changes.add(name)
        continue
    if name == 'MANIFEST.sha256':
        if changes.get(name) != {'before': before, 'after': 'REGENERATED_VERIFIED_SEPARATELY'}:
            raise ValueError('manifest regeneration must be declared')
        actual_changes.add(name)
        for line in path.read_text().splitlines():
            expected, relative = line.split('  ', 1)
            file = root/relative
            if file.is_symlink() or not file.resolve().is_relative_to(root) or not file.is_file():
                raise ValueError('manifest file boundary')
            with file.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('manifest hash mismatch: '+relative)
        continue
    with path.open('rb') as stream:
        after = hashlib.file_digest(stream, 'sha256').hexdigest()
    if before != after:
        actual_changes.add(name)
        if changes.get(name) != {'before': before, 'after': after}:
            raise ValueError('unaccounted historical drift: '+name)
if set(changes) != actual_changes:
    raise ValueError('stale repair accounting')
print(json.dumps({'inherited_files': len(freeze['files']), 'unchanged_files': len(freeze['files'])-len(changes), 'exact_accounted_repairs': sorted(changes), 'qualification_status': 'NOT_EVALUATED'}))
