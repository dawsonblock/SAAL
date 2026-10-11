"""Real child TLS service, not a mocked HTTP transport."""
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from saal_rsi.attempt_authority import Witness, CustodianMirror, evaluate_once
from saal_rsi.attempt_service import TLSClient
from saal_rsi.spec import sha


def certificates(root):
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'ephemeral-test-ca')])
    now = datetime.now(timezone.utc)
    ca = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True, path_length=0), True).add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False).add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False).add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False), True).sign(ca_key, hashes.SHA256())
    (root / 'ca.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    pins = {}
    for role in ('server', 'custodian_server', 'custodian', 'evaluator', 'controller', 'worker'):
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, role)])
        is_server = role.endswith('server')
        builder = x509.CertificateBuilder().subject_name(subject).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1)).add_extension(x509.BasicConstraints(ca=False, path_length=None), True).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH if is_server else ExtendedKeyUsageOID.CLIENT_AUTH]), False)
        if is_server: builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), False)
        builder = builder.add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False).add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False).add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False), True)
        certificate = builder.sign(ca_key, hashes.SHA256())
        (root / f'{role}.pem').write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        private = root / f'{role}.key'; private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())); private.chmod(0o600)
        pins[role] = certificate.fingerprint(hashes.SHA256()).hex()
    return pins


@pytest.fixture
def service(tmp_path):
    pins = certificates(tmp_path); signing = ed25519.Ed25519PrivateKey.generate()
    private = tmp_path / 'witness.key'; private.write_bytes(signing.private_bytes_raw()); private.chmod(0o600)
    enrollment = {'tenant': 'process_test', 'family': 'fixed', 'question_digest': sha('research_question'), 'total_alpha': .05, 'max_attempts': 5}
    witness = Witness.initialize(tmp_path / 'witness.db', signing, enrollment)
    server = {'host': '127.0.0.1', 'port': 0, 'ca': str(tmp_path / 'ca.pem'), 'certificate': str(tmp_path / 'server.pem'), 'tls_key': str(tmp_path / 'server.key'), 'clients': {'custodian': pins['custodian'], 'evaluator': pins['evaluator']}}
    config = tmp_path / 'config.json'; config.write_text(json.dumps({'enrollment': enrollment, 'signing_key': str(private), 'server': server}))
    root = Path(__file__).resolve().parents[2]
    environment = {'PATH': os.environ['PATH'], 'PYTHONPATH': str(root / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    child = subprocess.Popen([sys.executable, '-m', 'saal_rsi.attempt_service', 'serve-witness', '--config', str(config), '--database', str(witness.path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, 'child witness did not start'
        port = json.loads(child.stdout.readline())['address'][1]
        def client(role):
            return TLSClient({'url': f'https://localhost:{port}', 'ca': str(tmp_path / 'ca.pem'), 'certificate': str(tmp_path / f'{role}.pem'), 'tls_key': str(tmp_path / f'{role}.key'), 'server_certificate_sha256': pins['server']})
        yield witness, client, tmp_path, pins
    finally:
        child.terminate(); child.communicate(timeout=10)


def test_real_mtls_reservation_consumption_and_worker_denial(service):
    witness, client, root, _ = service
    mirror = CustodianMirror.initialize(root / 'mirror.db', client('custodian'), witness.enrollment, witness.pin)
    bindings = {k: sha(k) for k in ('candidate_digest', 'baseline_digest', 'protocol_digest', 'holdout_digest', 'sampling_review_digest', 'question_digest')}; bindings['question_digest'] = witness.enrollment['question_digest']
    for role in ('worker', 'controller'):
        with pytest.raises(PermissionError): client(role)('head', {})
    reservation = mirror.reserve(bindings)
    result = evaluate_once(reservation, bindings, client('evaluator'), witness.pin, witness.enrollment, lambda alpha: {'actual_inference_entered_after_consumption': witness.head()['payload']['seq'] == 2, 'alpha': alpha})
    assert result['measurement']['actual_inference_entered_after_consumption']
    mirror.acknowledge_consumption(result['consumption'])
    with pytest.raises(PermissionError): evaluate_once(reservation, bindings, client('evaluator'), witness.pin, witness.enrollment, lambda _: pytest.fail('replayed receipt entered inference'))
    bindings['candidate_digest'] = sha('next'); assert mirror.reserve(bindings)['payload']['alpha'] == .0125


def test_real_mtls_server_identity_substitution(service):
    witness, client, _, _ = service
    c = client('custodian'); c.config['server_certificate_sha256'] = '0' * 64
    with pytest.raises(PermissionError): c('head', {})


def test_no_insecure_transport():
    with pytest.raises(ValueError): TLSClient({'url': 'http://localhost', 'ca': '', 'certificate': '', 'tls_key': '', 'server_certificate_sha256': '0' * 64})


def test_controller_can_only_reserve_through_real_custodian_service(service):
    witness, client, root, pins = service
    mirror_path = root/'custodian.db'
    CustodianMirror.initialize(mirror_path,client('custodian'),witness.enrollment,witness.pin)
    server={'host':'127.0.0.1','port':0,'ca':str(root/'ca.pem'),'certificate':str(root/'custodian_server.pem'),'tls_key':str(root/'custodian_server.key'),'clients':{'controller':pins['controller'],'evaluator':pins['evaluator']}}
    cfg=root/'custodian-config.json';cfg.write_text(json.dumps({'server':server,'witness_client':client('custodian').config,'enrollment':witness.enrollment,'witness_public_key':witness.pin}))
    source=Path(__file__).resolve().parents[2]
    child=subprocess.Popen([sys.executable,'-m','saal_rsi.attempt_service','serve-custodian','--config',str(cfg),'--database',str(mirror_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={'PATH':os.environ['PATH'],'PYTHONPATH':str(source/'src'),'PYTHONDONTWRITEBYTECODE':'1'})
    try:
        ready,_,_=select.select([child.stdout],[],[],10);assert ready
        port=json.loads(child.stdout.readline())['address'][1]
        def custody(role):return TLSClient({'url':f'https://localhost:{port}','ca':str(root/'ca.pem'),'certificate':str(root/f'{role}.pem'),'tls_key':str(root/f'{role}.key'),'server_certificate_sha256':pins['custodian_server']})
        bindings={k:sha(k) for k in ('candidate_digest','baseline_digest','protocol_digest','holdout_digest','sampling_review_digest','question_digest')};bindings['question_digest']=witness.enrollment['question_digest']
        with pytest.raises(PermissionError):custody('worker')('reserve',{'bindings':bindings})
        with pytest.raises(PermissionError):custody('evaluator')('reserve',{'bindings':bindings})
        reservation=custody('controller')('reserve',{'bindings':bindings})
        result=evaluate_once(reservation,bindings,client('evaluator'),witness.pin,witness.enrollment,lambda alpha:{'measured':True})
        assert custody('evaluator')('acknowledge',{'consumption':result['consumption']})=={'acknowledged':True}
        bindings['candidate_digest']=sha('new');assert custody('controller')('reserve',{'bindings':bindings})['payload']['alpha']==.0125
    finally:
        child.terminate();child.communicate(timeout=10)
