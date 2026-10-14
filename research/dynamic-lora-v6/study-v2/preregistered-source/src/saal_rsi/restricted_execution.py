"""Optional macOS process restrictions, never a cross-administrator guarantee.

Absent OS support fails closed. No controller/evaluator credentials are inherited.
This protects declared paths and denies networking/writes outside the output;
it is not a complete hostile-code container or an independent evaluator host.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def restricted_process(argv, output, protected_roots, *, timeout=60):
    tool = shutil.which('sandbox-exec') if sys.platform == 'darwin' else None
    if tool is None:
        raise PermissionError('macOS restriction unavailable; no unconfined fallback')
    if type(argv) is not list or not argv or any(type(a) is not str or not a for a in argv) or type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError('bounded executable arguments')
    output = Path(output).resolve()
    if not output.is_dir() or not protected_roots:
        raise ValueError('explicit output and protected roots required')
    roots = [Path(p).resolve() for p in protected_roots]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        raise ValueError('output overlaps protected authority storage')
    quote = lambda p: json.dumps(str(p), ensure_ascii=False)
    profile = ['(version 1)', '(allow default)', '(deny network*)', '(deny file-write*)',
               f'(allow file-write* (subpath {quote(output)}))']
    profile.extend(f'(deny file-read* (subpath {quote(root)}))' for root in roots)
    environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'TMPDIR': str(output),
                   'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    return subprocess.run([tool, '-p', '\n'.join(profile), *argv], cwd=output, env=environment,
                          capture_output=True, text=True, timeout=timeout, check=False)
