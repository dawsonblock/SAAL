"""Bounded mTLS transport for the witness and custodian.

CA validation AND an exact client-certificate fingerprint select the role.
Only the witness has the witness signing key; no key is delivered in replies.
Separate machines/OS administrators are needed for operational independence.
There is no insecure HTTP or token-only fallback.
"""
import argparse
import hashlib
import http.client
import json
import ssl
import stat
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .attempt_authority import Witness, CustodianMirror
from .spec import canonical, parse_unique_json

MAX_BODY = 64 * 1024


def private_key(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_size != 32:
            raise PermissionError('service signing key must be private raw Ed25519')
        return Ed25519PrivateKey.from_private_bytes(os.read(fd, 33))
    finally:
        os.close(fd)


class TLSClient:
    def __init__(self, config):
        if type(config) is not dict or set(config) != {'url', 'ca', 'certificate', 'tls_key', 'server_certificate_sha256'}:
            raise ValueError('mTLS client configuration')
        parsed = urlsplit(config['url'])
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.path not in ('', '/') or parsed.query or parsed.fragment or not parsed.hostname:
            raise ValueError('HTTPS service origin only')
        if type(config['server_certificate_sha256']) is not str or len(config['server_certificate_sha256']) != 64:
            raise ValueError('exact server certificate pin required')
        self.config, self.host, self.port = config, parsed.hostname, parsed.port or 443
        self.context = ssl.create_default_context(cafile=config['ca'])
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.context.load_cert_chain(config['certificate'], config['tls_key'])

    def __call__(self, operation, arguments):
        request = canonical({'operation': operation, 'arguments': arguments})
        if len(request) > MAX_BODY:
            raise ValueError('RPC request bound')
        connection = http.client.HTTPSConnection(self.host, self.port, timeout=10, context=self.context)
        try:
            connection.connect()
            actual = hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest()
            if actual != self.config['server_certificate_sha256']:
                raise PermissionError('service certificate substitution')
            connection.request('POST', '/rpc', body=request, headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            raw = response.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY or response.status != 200:
                raise PermissionError('authority unavailable or request rejected; no fallback')
            body = parse_unique_json(raw)
            if set(body) != {'result'}:
                raise ValueError('RPC response shape')
            return body['result']
        finally:
            connection.close()


def serve(config, dispatch):
    fields = {'host', 'port', 'ca', 'certificate', 'tls_key', 'clients'}
    if type(config) is not dict or set(config) != fields or type(config['clients']) is not dict or not config['clients']:
        raise ValueError('server configuration shape')
    if len(config['clients']) != len(set(config['clients'].values())):
        raise ValueError('each client role needs a distinct certificate')
    if any(role not in ('controller', 'custodian', 'evaluator') or type(pin) is not str or len(pin) != 64 for role, pin in config['clients'].items()):
        raise ValueError('service role pins')
    roles = {fingerprint: role for role, fingerprint in config['clients'].items()}
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=config['ca'])
    context.load_cert_chain(config['certificate'], config['tls_key'])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log bodies, certificate material, credentials or holdout identity.

        def do_POST(self):
            try:
                self.connection.settimeout(10)
                fingerprint = hashlib.sha256(self.connection.getpeercert(binary_form=True)).hexdigest()
                role = roles.get(fingerprint)
                if role is None or self.path != '/rpc' or self.headers.get('Transfer-Encoding') or self.headers.get_all('Content-Length') is None or len(self.headers.get_all('Content-Length')) != 1:
                    raise PermissionError('unauthorized or ambiguous request')
                length = int(self.headers['Content-Length'])
                if not 1 <= length <= MAX_BODY:
                    raise ValueError('RPC request length')
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('incomplete request')
                body = parse_unique_json(raw)
                if set(body) != {'operation', 'arguments'} or type(body['operation']) is not str or type(body['arguments']) is not dict:
                    raise ValueError('RPC request shape')
                response = canonical({'result': dispatch(role, body['operation'], body['arguments'])})
                if len(response) > MAX_BODY:
                    raise ValueError('RPC response bound')
                self.send_response(200)
            except Exception:
                response = b'{"error":"authority request denied"}'
                self.send_response(403)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def get_request(self):
            socket, address = super().get_request()
            socket.settimeout(10)
            try:
                return context.wrap_socket(socket, server_side=True), address
            except Exception:
                socket.close()
                raise

    with Server((config['host'], config['port']), Handler) as server:
        print(json.dumps({'service': 'ready', 'address': server.server_address}), flush=True)
        server.serve_forever(poll_interval=.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('init-witness', 'serve-witness', 'init-custodian', 'serve-custodian'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--database', required=True)
    args = parser.parse_args()
    config = parse_unique_json(Path(args.config).read_bytes())
    if args.mode.endswith('witness'):
        key = private_key(config['signing_key'])
        if args.mode == 'init-witness':
            Witness.initialize(args.database, key, config['enrollment'])
        else:
            witness = Witness(args.database, key, config['enrollment'])
            serve(config['server'], witness.transact)
    else:
        client = TLSClient(config['witness_client'])
        if args.mode == 'init-custodian':
            CustodianMirror.initialize(args.database, client, config['enrollment'], config['witness_public_key'])
        else:
            mirror = CustodianMirror(args.database, client, config['enrollment'], config['witness_public_key'])

            def dispatch(role, operation, arguments):
                if role == 'controller' and operation == 'reserve' and set(arguments) == {'bindings'}:
                    return mirror.reserve(arguments['bindings'])
                if role == 'evaluator' and operation == 'acknowledge' and set(arguments) == {'consumption'}:
                    mirror.acknowledge_consumption(arguments['consumption'])
                    return {'acknowledged': True}
                raise PermissionError('custody service role or operation denied')

            serve(config['server'], dispatch)


if __name__ == '__main__':
    main()
