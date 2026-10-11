"""Unix socket demonstration of an OS-process-separated authority decision service.

Use a distinct OS identity and segregated permissions. This server authorizes
only pre-installed delegations (never client-submitted delegations). It does not
execute external tools. Human approval is an intentionally out-of-band control.
"""
import json
import os
import socket
import socketserver
import stat
import struct
from collections.abc import Mapping
from pathlib import Path
from .authority import Authority
from .types import Action, Delegation

MAX_LINE = 16_384


def secure_key(path: str) -> bytes:
    """Read a private regular key atomically through its opened descriptor.

    O_NOFOLLOW and fstat defeat path replacement between lstat and read.
    There is deliberately no fallback on platforms lacking O_NOFOLLOW.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("secure key loading unsupported on this platform")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise PermissionError("secure key open refused") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError("key must be a regular file")
        if st.st_uid != os.geteuid():
            raise PermissionError("key must be owned by the authority identity")
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError("key file must not grant access to group/others")
        if not 32 <= st.st_size <= 4096:
            raise ValueError("key length outside trusted limits")
        with os.fdopen(fd, "rb", closefd=False) as file:
            return file.read()
    finally:
        os.close(fd)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = val
    return result


def _peer_uid(connection) -> int:
    """Require kernel-reported peer UID. No client-supplied identity fallback."""
    if hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                    struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", raw)
        return uid
    if hasattr(connection, "getpeereid"):
        uid, _gid = connection.getpeereid()
        return uid
    # CPython on macOS may omit socket.getpeereid; libc still exposes the
    # kernel-backed API. Fail closed if it returns an error.
    import sys
    if sys.platform == 'darwin':
        import ctypes
        libc=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
        getpeereid=libc.getpeereid
        getpeereid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)]
        getpeereid.restype=ctypes.c_int
        uid=ctypes.c_uint();gid=ctypes.c_uint()
        if getpeereid(connection.fileno(),ctypes.byref(uid),ctypes.byref(gid))!=0:
            raise PermissionError('kernel peer identity lookup failed')
        return uid.value
    raise PermissionError("peer identity unavailable on this platform")


def load_peer_bindings(path: str) -> dict[int, frozenset[tuple[str, str]]]:
    """Trusted administrator-owned mapping UID -> authorized principal/task pairs.

    This binds requests to a Unix process identity, not to an authenticated
    human. Never share an actor UID with evaluator, signer or authority.
    """
    key = secure_key(path)
    data = json.loads(key, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, dict) or not data:
        raise ValueError("peer bindings must be nonempty UID map")
    result = {}
    for uid, scopes in data.items():
        if not isinstance(uid, str) or not uid.isdecimal() or str(int(uid)) != uid:
            raise ValueError("invalid peer uid")
        if not isinstance(scopes, list) or not scopes:
            raise ValueError("empty uid bindings")
        allowed = set()
        for item in scopes:
            if (not isinstance(item, dict) or set(item) != {"principal", "task_id"}
                    or not all(isinstance(x, str) and x for x in item.values())):
                raise ValueError("invalid binding")
            allowed.add((item["principal"], item["task_id"]))
        if len(allowed) != len(scopes):
            raise ValueError("duplicate principal/task binding")
        result[int(uid)] = frozenset(allowed)
    return result


class Handler(socketserver.StreamRequestHandler):
    def setup(self):
        # Limit incomplete request / slow-client occupancy of the serialized server.
        self.request.settimeout(3.0)
        super().setup()

    def handle(self):
        try:
            raw = self.rfile.readline(MAX_LINE + 1)
        except (TimeoutError, OSError):
            return
        if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
            response = {"status": "ERROR", "reason": "invalid request size"}
        else:
            try:
                req = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
                if set(req) != {"action"} or not isinstance(req["action"], dict):
                    raise ValueError("invalid request shape")
                if set(req["action"]) != {"tool", "args", "principal", "task_id"}:
                    raise ValueError("invalid action schema")
                action = Action(**req["action"])
                peer_uid = _peer_uid(self.request)
                allowed = self.server.peer_bindings.get(peer_uid, frozenset())
                if (action.principal, action.task_id) not in allowed:
                    response = {"status": "DENY", "reasons": ["peer identity not authorized for delegation"]}
                    self.wfile.write(json.dumps(response, sort_keys=True).encode() + b"\n")
                    return
                delegation = self.server.delegations.get((action.principal, action.task_id))
                if delegation is None:
                    response = {"status": "DENY", "reasons": ["delegation not installed"]}
                else:
                    dec = self.server.authority.decide(action, delegation)
                    response = {"status": dec.disposition, "risk": dec.risk,
                                "reasons": list(dec.reasons), "token": dec.token}
            except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError, OSError, PermissionError):
                response = {"status": "DENY", "reasons": ["malformed request"]}
        self.wfile.write(json.dumps(response, sort_keys=True).encode() + b"\n")


class GateServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, path: str, authority: Authority, delegations: list[Delegation],
                 *, peer_bindings: Mapping[int, frozenset[tuple[str, str]]],
                 socket_group_gid: int | None = None):
        self.authority = authority
        if not peer_bindings:
            raise ValueError("explicit peer UID authorization required")
        self.peer_bindings = {int(uid): frozenset(pairs)
                              for uid, pairs in peer_bindings.items()}
        self.delegations = {(d.principal, d.task_id): d for d in delegations}
        # Never unlink an existing socket: it might belong to a live or more
        # privileged service. Only an authenticated operator may remove stale
        # socket nodes before startup. Refuse symlinks, including dangling ones.
        p = Path(path)
        if p.is_symlink():
            raise PermissionError("socket path must not be symlink")
        if p.exists():
            raise FileExistsError("socket already exists; operator must inspect and remove it")
        oldmask = os.umask(0o177)
        try:
            super().__init__(path, Handler)
        finally:
            os.umask(oldmask)
        if socket_group_gid is not None:
            os.chown(path, -1, socket_group_gid)
            os.chmod(path, 0o660)
        else:
            os.chmod(path, 0o600)

    def server_close(self):
        socket = self.server_address
        super().server_close()
        try:
            Path(socket).unlink(missing_ok=True)
        except OSError:
            pass


def load_delegations(path: str) -> list[Delegation]:
    # Delegation files are authority-owned policy inputs, not actor-owned JSON.
    data = json.loads(secure_key(path), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, list):
        raise ValueError("delegations must be a list")
    delegates = [Delegation(**{**row, "allowed_tools": tuple(row["allowed_tools"])}) for row in data]
    if len({(d.principal, d.task_id) for d in delegates}) != len(delegates):
        raise ValueError("duplicate identity in authority-side delegation file")
    return delegates
