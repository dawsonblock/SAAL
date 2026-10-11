import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from saal.authority import Authority
from saal.gate_server import GateServer, secure_key, load_peer_bindings
from saal.types import Delegation


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix domain socket required")
class GateServerTests(unittest.TestCase):
    def test_only_installed_delegations_and_predefined_tools(self):
        with tempfile.TemporaryDirectory() as td:
            sock = str(Path(td)/"authority.sock")
            authority = Authority(b"a"*32, b"b"*32)
            d = Delegation("joe", "task", "read", ("schedule.read",), 2_000_000_000)
            with GateServer(sock, authority, [d], peer_bindings={os.getuid(): frozenset({("joe", "task")})}) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                def ask(tool, user="joe", task="task"):
                    message = {"action": {"tool": tool, "args": {"calendar_id": "demo"} if tool == "schedule.read" else {},
                                          "principal": user, "task_id": task}}
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(sock)
                        client.sendall((json.dumps(message)+"\n").encode())
                        return json.loads(client.makefile("rb").readline())
                self.assertEqual(ask("schedule.read")["status"], "ALLOW")
                self.assertEqual(ask("schedule.read", user="intruder")["status"], "DENY")
                self.assertEqual(ask("model.promote")["status"], "DENY")
                self.assertEqual(ask("system.change_policy")["status"], "DENY")
                server.shutdown()
                thread.join(timeout=5)
            authority.close()


    def test_refuses_to_replace_existing_active_service_socket(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "existing.sock")
            listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listening.bind(path)
            listening.listen(1)
            authority = Authority(b"k"*32, b"h"*32)
            try:
                with self.assertRaises(FileExistsError):
                    GateServer(path, authority, [], peer_bindings={os.getuid(): frozenset({("joe", "task")})})
                self.assertTrue(Path(path).exists())
            finally:
                listening.close()
                authority.close()

    def test_refuses_to_replace_symlinked_socket(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "target.sock"
            link = Path(td) / "link.sock"
            link.symlink_to(target)
            authority = Authority(b"k"*32, b"h"*32)
            try:
                with self.assertRaises(PermissionError):
                    GateServer(str(link), authority, [], peer_bindings={os.getuid(): frozenset({("joe", "task")})})
            finally:
                authority.close()

    def test_key_symlink_denied_even_if_target_private(self):
        with tempfile.TemporaryDirectory() as td:
            key = Path(td) / "private.key"
            key.write_bytes(b"x" * 32)
            os.chmod(key, 0o600)
            link = Path(td) / "link.key"
            link.symlink_to(key)
            with self.assertRaises(PermissionError):
                secure_key(str(link))

    def test_rejects_world_readable_keys(self):
        if os.name == "nt":
            self.skipTest("POSIX key permission checks")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"key"
            p.write_bytes(b"a"*32)
            os.chmod(p, 0o644)
            with self.assertRaises(PermissionError):
                secure_key(str(p))
