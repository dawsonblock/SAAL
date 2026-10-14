"""Private local SQLite preflight; custody and parent-directory protection still required."""
from pathlib import Path
import os
import stat


def private_database(path):
    path=Path(path)
    if path.is_symlink():raise PermissionError('database symlink forbidden')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|getattr(os,'O_NONBLOCK',0),0o600)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):raise PermissionError('database must be regular file')
        if info.st_uid!=os.geteuid() or info.st_nlink!=1:raise PermissionError('database owner or hardlink mismatch')
        if stat.S_IMODE(info.st_mode)&0o077:raise PermissionError('database must be private')
    finally:os.close(fd)
    return path
