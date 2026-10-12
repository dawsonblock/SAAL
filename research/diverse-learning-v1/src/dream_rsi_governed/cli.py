from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .canonical import sha256_digest
from .canary import compare_live_canary
from .datasets import DatasetSplit, partition
from .demo import make_demo_worlds
from .evaluator import ReplayEvaluator
from .ledger import AppendOnlyLedger
from .models import CandidatePolicy
from .policy import FixedBreadthPolicy, RiskAwareUCBPolicy
from .qualification import Qualifier
from .replay import ReplayEngine
from .signing import generate_private_key, private_key_bytes
from .promotion import PromotionAuthority


def _dump(obj):
    print(json.dumps(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj, indent=2, sort_keys=True))


def cmd_demo(args):
    if args.promote:
        raise SystemExit("Legacy --promote is disabled; use the independent SAAL-RSI research activation path")
    worlds = make_demo_worlds(args.worlds, args.seed)
    parts = partition(worlds)
    baseline = FixedBreadthPolicy()
    candidate = RiskAwareUCBPolicy(beta=args.beta)
    evaluator = ReplayEvaluator(ReplayEngine(seed=args.seed))
    # Ensure the demo can run even if hash partition happens to create tiny holdout.
    holdout = parts[DatasetSplit.HOLDOUT]
    selection = parts[DatasetSplit.SELECTION]
    if len(holdout) < 3:
        holdout = worlds[-max(3, len(worlds)//5):]
    if not selection:
        selection = worlds[:max(1, len(worlds)//5)]
    meta = CandidatePolicy(candidate.policy_id, "1", sha256_digest({"policy": candidate.policy_id, "beta": args.beta}), {"beta": args.beta}, baseline.policy_id)
    qualifier = Qualifier(evaluator, min_holdout_worlds=3, max_score_regression=args.max_score_regression)
    rec = qualifier.qualify(meta, candidate, baseline, selection, holdout)
    _dump(rec)


def main():
    p = argparse.ArgumentParser(prog="dream-rsi", description="Governed Dream-RSI reference runtime")
    sub = p.add_subparsers(required=True)
    d = sub.add_parser("demo", help="run a deterministic end-to-end qualification demo")
    d.add_argument("--worlds", type=int, default=30)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--beta", type=float, default=1.0)
    d.add_argument("--max-score-regression", type=float, default=0.0)
    d.add_argument("--promote", action="store_true")
    d.add_argument("--out")
    d.set_defaults(func=cmd_demo)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
