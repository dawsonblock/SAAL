"""Custodian-owned signed preregistration and one-shot fresh-holdout ledger.

Custodian must run under a separate UID/host and publish receipts outside evaluator
control for independent anchoring. Local demo enforces order, not operator honesty.
"""
import base64
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import sqlite3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .spec import canonical, sha
from .identity import require_identity
from .validation import public_rules


def public_hex(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()


def verify_receipt(receipt, pinned_public_key):
    from .validation import key_bytes,validate_rules
    if type(receipt)is not dict or set(receipt)!={'payload','signature_b64'}:
        raise ValueError('receipt shape')
    payload=receipt['payload']
    fields={'schema','plan_digest','proposal_digest','identity','config_digest','family','attempt',
            'alpha','custodian_key_sha256','anchored_at','evaluation_rules'}
    if type(payload)is not dict or set(payload)!=fields or payload['schema']!='saal-rsi-anchor-v3':
        raise ValueError('receipt payload schema')
    raw=key_bytes(pinned_public_key)
    if payload['custodian_key_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('custodian identity mismatch')
    if payload['family']!='saal-rsi-research-v2':raise ValueError('unknown protocol family')
    if type(payload['attempt'])is not int or not 1<=payload['attempt']<=10:
        raise ValueError('invalid alpha attempt')
    if type(payload['alpha'])is not float or payload['alpha']!=0.05/(2**payload['attempt']):
        raise ValueError('invalid alpha allocation')
    validate_rules(payload['evaluation_rules'])
    if payload['config_digest']!=sha(payload['evaluation_rules']['config']):raise ValueError('config commitment mismatch')
    if type(payload['identity'])is not dict:raise ValueError('identity schema')
    for name in ('plan_digest','proposal_digest','config_digest'):
        digest=payload[name]
        if type(digest)is not str or not digest.startswith('sha256:') or len(digest)!=71:
            raise ValueError('invalid digest')
        try:bytes.fromhex(digest[7:])
        except ValueError as exc:raise ValueError('invalid digest encoding') from exc
    try:
        timestamp=datetime.fromisoformat(payload['anchored_at'])
        if timestamp.tzinfo is None:raise ValueError('missing timezone')
        Ed25519PublicKey.from_public_bytes(raw).verify(
            base64.b64decode(receipt['signature_b64'],validate=True),canonical(payload))
    except Exception as exc:raise ValueError('invalid custodian signature or timestamp') from exc


class Registry:
    def __init__(self, path, pinned_public_key):
        self.path=Path(path)
        if self.path.is_symlink():raise PermissionError('registry symlink forbidden')
        from .validation import key_bytes
        pinned_public_key=key_bytes(pinned_public_key).hex()
        self.public_key=pinned_public_key
        from .storage import private_database
        private_database(self.path)
        self.db=sqlite3.connect(str(self.path),timeout=10)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS trust (key TEXT PRIMARY KEY)')
        self.db.execute('CREATE TABLE IF NOT EXISTS anchors (digest TEXT PRIMARY KEY, receipt BLOB NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)')
        self.db.execute('CREATE TABLE IF NOT EXISTS seeds (seed INTEGER PRIMARY KEY, anchor TEXT NOT NULL REFERENCES anchors(digest))')
        self.db.execute('CREATE TABLE IF NOT EXISTS families (family TEXT PRIMARY KEY, attempts INTEGER NOT NULL)')
        try:
            with self.db:
                rows=self.db.execute('SELECT key FROM trust').fetchall()
                if rows and rows!=[(pinned_public_key,)]:raise ValueError('registry trust key changed')
                if not rows:self.db.execute('INSERT INTO trust VALUES (?)',(pinned_public_key,))
        except BaseException:
            self.db.close();raise


    def anchor(self, proposal, plan, custodian_key, *, family='saal-rsi-research-v2'):
        from .assurance import validate_proposal, validate_plan
        validate_proposal(proposal);validate_plan(plan)
        if public_hex(custodian_key)!=self.public_key:raise PermissionError('custodian key mismatch')
        if family!='saal-rsi-research-v2':raise ValueError('unknown preregistered family')
        if set(plan['seeds']) & (set(proposal['development_seeds'])|set(proposal['selection_seeds'])):
            raise ValueError('holdout overlaps research')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT attempts FROM families WHERE family=?',(family,)).fetchone()
            attempt=(row[0] if row else 0)+1
            if attempt>10:raise ValueError('alpha schedule exhausted; new scientific protocol required')
            # Alpha allocated when anchored, even if evaluation crashes or is never run.
            payload={'schema':'saal-rsi-anchor-v3','plan_digest':sha(plan),
                     'proposal_digest':proposal['proposal_digest'],'identity':plan['execution_identity'],
                     'config_digest':sha(plan['config']),'family':family,'attempt':attempt,
                     'alpha':0.05/(2**attempt),
                     'custodian_key_sha256':hashlib.sha256(bytes.fromhex(self.public_key)).hexdigest(),
                     'evaluation_rules':public_rules(plan),
                     'anchored_at':datetime.now(timezone.utc).isoformat()}
            receipt={'payload':payload,'signature_b64':base64.b64encode(custodian_key.sign(canonical(payload))).decode('ascii')}
            digest=sha(receipt)
            self.db.execute('INSERT INTO anchors VALUES (?,?,0)',(digest,canonical(receipt)))
            for seed in plan['seeds']:self.db.execute('INSERT INTO seeds VALUES (?,?)',(seed,digest))
            self.db.execute('INSERT INTO families VALUES (?,?) ON CONFLICT(family) DO UPDATE SET attempts=excluded.attempts',(family,attempt))
            self.db.commit()
            return receipt
        except BaseException:
            self.db.rollback();raise

    def consume(self, receipt, proposal, plan):
        verify_receipt(receipt,self.public_key)
        payload=receipt['payload']
        from .validation import public_rules
        if payload['evaluation_rules']!=public_rules(plan):raise ValueError('evaluation rules commitment changed')
        require_identity(payload['identity'])
        if payload['plan_digest']!=sha(plan) or payload['proposal_digest']!=proposal['proposal_digest'] or payload['config_digest']!=sha(plan['config']):
            raise ValueError('anchor commitment mismatch')
        if payload['identity']!=plan['execution_identity']:raise ValueError('anchor identity mismatch')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT receipt,consumed FROM anchors WHERE digest=?',(sha(receipt),)).fetchone()
            if not row or bytes(row[0])!=canonical(receipt) or row[1]:raise ValueError('unknown or consumed holdout anchor')
            seeds=[r[0] for r in self.db.execute('SELECT seed FROM seeds WHERE anchor=?',(sha(receipt),))]
            if sorted(seeds)!=sorted(plan['seeds']):raise ValueError('registry holdout identity mismatch')
            self.db.execute('UPDATE anchors SET consumed=1 WHERE digest=?',(sha(receipt),))
            self.db.commit()
        except BaseException:
            self.db.rollback();raise
        return payload['alpha']

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
