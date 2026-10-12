"""Only activation entry point for the integrated package; external activation denied."""
import base64
from pathlib import Path
import sqlite3
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .spec import canonical,sha
from .assurance import verify_attestation,validate_proposal

class ActivationDenied(PermissionError):pass


def authorize_external_activation(*args,**kwargs):
    raise ActivationDenied('External activation prohibited: research prototype lacks production qualification')


def authorize_research_activation(proposal, envelope, approval, *, evaluator_public_key,
                                  anchor_public_key, human_public_key, replay_db, now=None):
    """Return a frozen Candidate for simulation after independently signed exact approval.

    Custody/permissions and trusted calling runtime remain deployment obligations.
    Never returns authority to execute external tools or admit arbitrary code/weights.
    """
    from .validation import key_bytes
    try:
        role_keys={key_bytes(value) for value in (evaluator_public_key,anchor_public_key,human_public_key)}
    except ValueError as exc:raise ActivationDenied('invalid role key') from exc
    if len(role_keys)!=3:
        raise ActivationDenied('three distinct authority keys required')
    if not verify_attestation(envelope,evaluator_public_key,proposal=proposal,anchor_public_key=anchor_public_key):
        raise ActivationDenied('invalid qualification')
    report=envelope['report']
    if report['qualification_passed'] is not True:raise ActivationDenied('qualification failed')
    try:
        if set(approval)!={'payload','signature_b64'}:raise ValueError('approval shape')
        payload=approval['payload']
        if set(payload)!={'schema','report_digest','candidate_digest','simulation_only','expires_at'}:raise ValueError('approval payload')
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(human_public_key)).verify(
            base64.b64decode(approval['signature_b64'],validate=True),canonical(payload))
        now=int(time.time()) if now is None else now
        if type(now)is not int or now<0:raise ValueError('invalid trusted simulation time')
        if (payload['schema']!='saal-rsi-human-approval-v2' or payload['simulation_only'] is not True
            or payload['report_digest']!=sha(envelope) or payload['candidate_digest']!=report['candidate_digest']
            or type(payload['expires_at'])is not int or payload['expires_at']<=now):
            raise ValueError('approval binding or expiry')
        from .storage import private_database
        path=private_database(replay_db)
        with sqlite3.connect(str(path)) as db:
            db.execute('CREATE TABLE IF NOT EXISTS approvals (digest TEXT PRIMARY KEY)')
            db.execute('INSERT INTO approvals VALUES (?)',(sha(approval),))
    except Exception as exc:
        raise ActivationDenied('invalid or already consumed human approval') from exc
    return validate_proposal(proposal)
