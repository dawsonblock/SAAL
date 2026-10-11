"""CRC, safe fresh extraction, complete hash inventory, then real tests in extraction."""
from pathlib import Path
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from verify_manifest import verify

with tempfile.TemporaryDirectory(prefix='saal-rsi-clean-') as directory:
    base=Path(directory).resolve()
    with zipfile.ZipFile(sys.argv[1]) as archive:
        if archive.testzip() is not None:raise ValueError('archive CRC failure')
        names=set()
        for entry in archive.infolist():
            path=(base/entry.filename).resolve()
            if entry.filename in names or not path.is_relative_to(base) or stat.S_ISLNK(entry.external_attr>>16):
                raise ValueError('unsafe archive entry')
            names.add(entry.filename)
        archive.extractall(base)
    roots=list(base.iterdir())
    if len(roots)!=1 or not roots[0].is_dir():raise ValueError('expected single package directory')
    root=roots[0]
    print('Verified manifest files:',verify(root),flush=True)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,env=env)
    if result.returncode:raise SystemExit(result.returncode)
    print('Clean-extraction tests passed; manifest after tests:',verify(root))
