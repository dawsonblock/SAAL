"""Standalone offline smoke test for v4's fully separated synthetic eval workflow.

Runs in a temporary directory and never publishes or deploys a model.
The purposely tiny prereg tests protocol correctness, not experimental rigor.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from saal.canonical import digest
from saal.evaluator_attestation import EvaluatorPublicKey, create_evaluator_keys
from saal.evaluation import authenticate_study, run_study
from saal.prereg import private_prereg
from saal.qualification import qualify
from saal.train import fit_intent_model


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        private, public = root/'evaluator.key', root/'evaluator.pub'
        create_evaluator_keys(private, public)
        prereg = private_prereg(10, 30, public)
        # Smoke-sized reproducibility check; not a preregistered independent study.
        prereg['episodes_per_seed'] = 3
        prereg['min_episodes'] = 30
        prereg['max_critical_event_rate_upper_bound'] = 1.0
        prereg['min_improvement_lower_bound'] = -100.0
        model = fit_intent_model(seed=1201, epochs=1, per_class=4)
        report = run_study(model, seeds=prereg['seeds'], episodes_per_seed=3,
                           prereg=prereg, eval_private_key_file=str(private))
        verifier = EvaluatorPublicKey(public)
        assert authenticate_study(report, verifier)
        result = qualify(report, verifier, prereg, model=model,
                         expected_prereg_digest=digest(prereg))
        assert result['replay_audit']['valid'], result
        assert not any('signature' in x or 'commitment' in x or 'evaluator' in x
                       for x in result['reasons']), result
        print('PASS: v4 Ed25519 evaluation, pinned public key, prereg, exact replay')
        print('Candidate decision:', result['verdict'], '(this smoke check does NOT justify promotion)')


if __name__ == '__main__':
    main()
