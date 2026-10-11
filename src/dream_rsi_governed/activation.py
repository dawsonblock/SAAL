from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .models import CanaryAttestation, PromotionManifest, QualificationRecord
from .promotion import PromotionAuthority


class ActivationDenied(RuntimeError):
    pass


def authorize_activation(
    manifest: PromotionManifest,
    qualification: QualificationRecord,
    canary: CanaryAttestation,
    signer_public_key: Ed25519PublicKey,
    observed_artifact_root_digest: str,
) -> bool:
    # v0.8: inherited signatures are archival evidence, never integrated activation.
    from saal_rsi.promotion import authorize_external_activation
    try:
        authorize_external_activation()
    except PermissionError as exc:
        raise ActivationDenied(str(exc)) from exc
