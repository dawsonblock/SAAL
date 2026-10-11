"""Entry point. Default demo uses no external APIs or actual external effects."""
import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

from . import __version__
from .agent import AssistanceActor
from .canonical import digest
from .authority import Authority, SimulatedExecutor
from .evaluation import run_study
from .evaluator_attestation import EvaluatorPublicKey, create_evaluator_keys
from .gate_server import GateServer, load_delegations, load_peer_bindings, secure_key
from .ledger import EvidenceLedger
from .runtime import run_simulated_session
from .qualification import qualify
from .replay import audit_replay
from .prereg import private_prereg, receipt
from .witness import create_witness_keys, create_witness_receipt
from .release import (attest_qualification, authorize_candidate, stage_candidate,
                      verify_candidate_authorization, verify_qualification)
from .sim import make_scenario
from .train import IntentClassifier, fit_intent_model
from .types import Action, Delegation


def write_json(path: str, value: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cmd_demo(_args):
    scenario = make_scenario(81013, 2)
    visible = scenario.public_view()
    actor = AssistanceActor(True, True)
    proposal = actor.propose(visible)
    if proposal["kind"] == "ASK":
        proposal = actor.after_answer(visible, scenario.answer_clarification())
    key = secrets.token_bytes(32)
    authority = Authority(key, secrets.token_bytes(32))
    delegate = Delegation("demo_user", "demo-1", "propose appointment",
                          ("schedule.propose",), int(time.time()) + 300)
    action = Action("schedule.propose", {"slot_id": proposal["slot_id"], "cost_cents": 0},
                    "demo_user", "demo-1")
    decision = authority.decide(action, delegate)
    execution = SimulatedExecutor(authority).execute(action, decision.token)
    authority.close()
    print(json.dumps({"scenario_public": visible, "proposal": proposal,
                      "authority": decision.disposition, "execution": execution,
                      "warning": "SIMULATION ONLY: no appointment was booked"}, indent=2))


def cmd_train(args):
    model = fit_intent_model(seed=args.seed, epochs=args.epochs)
    model.save(args.out)
    print(json.dumps({"status": "trained offline", "type": "linear_softmax (not LoRA)",
                      "output": args.out, "metadata": model.metadata}, indent=2))


def cmd_session(args):
    sc = make_scenario(args.seed, args.index)
    print(json.dumps(run_simulated_session(sc, args.directory,
                              consent_to_memory=args.consent_to_memory), indent=2))


def cmd_evaluate(args):
    model = IntentClassifier.load(args.model)
    if args.eval_key_file and args.eval_private_key:
        raise ValueError("select only one evaluator signing method")
    key = secure_key(args.eval_key_file) if args.eval_key_file else None
    prereg = json.loads(Path(args.prereg).read_text()) if args.prereg else None
    if (key is not None or args.eval_private_key) and prereg is None:
        raise ValueError("independently signed evaluation needs --prereg")
    seeds = args.seeds if args.seeds is not None else (
        prereg["seeds"] if prereg else [81013, 81017, 81019, 81023, 81031])
    episodes = args.episodes if args.episodes is not None else (
        prereg["episodes_per_seed"] if prereg else 120)
    witness_receipt = json.loads(Path(args.witness_receipt).read_text()) if args.witness_receipt else None
    result = run_study(model, seeds=seeds, episodes_per_seed=episodes,
                       eval_key=key, prereg=prereg,
                       eval_private_key_file=args.eval_private_key,
                       witness_receipt=witness_receipt,
                       witness_public_key=args.witness_public_key)
    write_json(args.out, result)
    summaries = {k: v["metrics"] for k, v in result["report"]["results"].items()}
    print(json.dumps({"output": args.out, "status": result["signing_status"],
                      "digest": result["report_digest"], "arms": summaries}, indent=2))


def cmd_qualify(args):
    report = json.loads(Path(args.report).read_text())
    prereg = json.loads(Path(args.prereg).read_text())
    if bool(args.eval_key_file) == bool(args.eval_public_key):
        raise ValueError("select exactly one of --eval-public-key or legacy --eval-key-file")
    key = (EvaluatorPublicKey(args.eval_public_key) if args.eval_public_key
           else secure_key(args.eval_key_file))
    model = IntentClassifier.load(args.model) if args.model else None
    witness_receipt = json.loads(Path(args.witness_receipt).read_text()) if args.witness_receipt else None
    outcome = qualify(report, key, prereg, model=model,
                      expected_prereg_digest=args.expected_prereg_digest,
                      witness_receipt=witness_receipt,
                      witness_public_key=args.witness_public_key)
    if args.out:
        write_json(args.out, outcome)
    if args.qualification_key_file:
        if not args.attestation_out or not args.reviewer:
            raise ValueError("qualification signing requires --attestation-out and --reviewer")
        envelope = attest_qualification(outcome, args.qualification_key_file, args.reviewer)
        write_json(args.attestation_out, envelope)
    print(json.dumps(outcome, indent=2))
    if outcome["verdict"] == "REJECT":
        return 3
    return 0


def cmd_candidate_authorize(args):
    qualification = json.loads(Path(args.qualification_attestation).read_text())
    prereg = json.loads(Path(args.prereg).read_text())
    result = authorize_candidate(qualification, args.qualifier_public_key, args.model,
                                 prereg, args.human_private_key, args.approver,
                                 args.rationale)
    write_json(args.out, result)
    print(json.dumps({"status": "SIGNED_STAGING_AUTHORIZATION", "output": args.out}, indent=2))


def cmd_candidate_stage(args):
    qualification = json.loads(Path(args.qualification_attestation).read_text())
    authorization = json.loads(Path(args.candidate_authorization).read_text())
    prereg = json.loads(Path(args.prereg).read_text())
    result = stage_candidate(authorization, args.human_public_key, qualification,
                             args.qualifier_public_key, args.model, prereg,
                             args.stage_root)
    print(json.dumps(result, indent=2))


def cmd_init_key(args):
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as fp:
        fp.write(secrets.token_bytes(32))
    print(f"Created restricted key file at {path}; DO NOT commit or share it")


def cmd_gate(args):
    akey = secure_key(args.authority_key_file)
    hkey = secure_key(args.human_key_file)
    authority = Authority(akey, hkey, replay_db=args.replay_db)
    delegations = load_delegations(args.delegations)
    try:
        with GateServer(args.socket, authority, delegations,
                        peer_bindings=load_peer_bindings(args.peer_bindings),
                        socket_group_gid=args.socket_group_gid) as server:
            print(f"Authority service started at {args.socket}; use a distinct OS user in deployment", flush=True)
            server.serve_forever(poll_interval=0.2)
    finally:
        authority.close()


def cmd_evaluator_keygen(args):
    create_evaluator_keys(args.private, args.public)
    print(json.dumps({"private": args.private, "public": args.public,
                      "public_key_sha256": EvaluatorPublicKey(args.public).fingerprint(),
                      "warning": "Private evaluator key must NEVER leave evaluator trust domain"}, indent=2))


def cmd_prereg_create(args):
    if not args.evaluator_public_key:
        raise ValueError("v4 prereg-create requires --evaluator-public-key")
    prereg = private_prereg(args.seeds, args.episodes, args.evaluator_public_key, args.witness_public_key)
    write_json(args.out, prereg)
    print(json.dumps({"output": args.out, **receipt(prereg),
                      "warning": "seeds are now revealed at this location; protect evaluation partition"}, indent=2))


def cmd_witness_keygen(args):
    create_witness_keys(args.private, args.public)
    print(json.dumps({"status": "INDEPENDENT_WITNESS_KEYS_CREATED",
                      "public_key_sha256": EvaluatorPublicKey(args.public).fingerprint()}, indent=2))


def cmd_prereg_witness(args):
    prereg = json.loads(Path(args.prereg).read_text())
    witness = create_witness_receipt(prereg, args.witness_private_key,
                                     args.witness_public_key, witness_identity=args.witness_identity)
    write_json(args.out, witness)
    print(json.dumps({"status": "WITNESSED_NOT_TIMESTAMP_PROVEN",
                      "receipt_digest": digest(witness), "out": args.out,
                      "warning": "Publish witness receipt to independent append-only log BEFORE evaluation"}, indent=2))


def cmd_ledger_demo(args):
    ledger = EvidenceLedger(args.db)
    ledger.append("delegation_issued", {"task": "example", "origin": "trusted_demo"})
    ledger.append("verified_outcome", {"verified": True, "kind": "sim_oracle"})
    summary = ledger.verify()
    ledger.close()
    if args.anchor:
        path = Path(args.anchor)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary["final_hash"] + "\n")
    print(json.dumps({**summary, "warning": "anchoring only becomes independent when stored out-of-band"}, indent=2))


def cmd_ledger_verify(args):
    ledger = EvidenceLedger(args.db)
    anchor = Path(args.anchor).read_text().strip() if args.anchor else None
    result = ledger.verify(anchor)
    ledger.close()
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 4


def main(argv=None):
    parser = argparse.ArgumentParser(prog="saal", description=f"SAAL v{__version__}: research prototype")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="safe toy assistant with one authorized simulated action")
    p = sub.add_parser("session-demo", help="integrated simulated session plus audit and optional memory")
    p.add_argument("--directory", default="artifacts/session")
    p.add_argument("--seed", type=int, default=81013)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--consent-to-memory", action="store_true")
    p = sub.add_parser("train", help="offline-train a small intent classifier")
    p.add_argument("--out", default="artifacts/intent-model.json")
    p.add_argument("--seed", type=int, default=1201)
    p.add_argument("--epochs", type=int, default=55)
    p = sub.add_parser("evaluate", help="run six-arm paired synthetic holdout experiment")
    p.add_argument("--model", default="artifacts/intent-model.json")
    p.add_argument("--out", default="artifacts/evaluation.json")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--eval-key-file", help="legacy v3 HMAC; do not use in v4")
    p.add_argument("--eval-private-key", help="v4 Ed25519 signer: restricted evaluator-only private key")
    p.add_argument("--prereg", help="frozen preregistration; mandatory for signed studies")
    p.add_argument("--witness-receipt", help="v5 preregistration witness receipt")
    p.add_argument("--witness-public-key", help="independent pinned witness verification key")
    p = sub.add_parser("qualify", help="verify evaluator signature and preregistered gates")
    p.add_argument("--report", required=True)
    p.add_argument("--prereg", default="configs/preregistered_eval.json")
    p.add_argument("--eval-key-file", help="legacy HMAC (v3 only)")
    p.add_argument("--eval-public-key", help="v4 Ed25519 public key, pinned by prereg")
    p.add_argument("--expected-prereg-digest", help="externally committed SHA-256; mandatory for v4/v5")
    p.add_argument("--witness-receipt", help="mandatory independently signed preregistration witness for v5")
    p.add_argument("--witness-public-key", help="independently pinned public key; mandatory for v5")
    p.add_argument("--model", help="exact candidate model; REQUIRED for v4 replay gate")
    p.add_argument("--out")
    p.add_argument("--qualification-key-file", help="qualifier-only Ed25519 private key")
    p.add_argument("--attestation-out")
    p.add_argument("--reviewer")
    p = sub.add_parser("candidate-authorize", help="human sign-off permits staging only")
    p.add_argument("--qualification-attestation", required=True)
    p.add_argument("--qualifier-public-key", required=True)
    p.add_argument("--human-private-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--approver", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("candidate-stage", help="verify both signatures and stage an immutable candidate")
    p.add_argument("--qualification-attestation", required=True)
    p.add_argument("--qualifier-public-key", required=True)
    p.add_argument("--human-public-key", required=True)
    p.add_argument("--candidate-authorization", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--stage-root", required=True)
    p = sub.add_parser("init-key", help="create secret file mode 0600")
    p.add_argument("--out", required=True)
    p = sub.add_parser("gate-server", help="Unix-socket authority; no tool execution")
    p.add_argument("--socket", required=True)
    p.add_argument("--authority-key-file", required=True)
    p.add_argument("--human-key-file", required=True)
    p.add_argument("--delegations", required=True)
    p.add_argument("--replay-db", required=True)
    p.add_argument("--peer-bindings", required=True,
                   help="root/authority-owned 0600 JSON UID-to-principal/task bindings")
    p.add_argument("--socket-group-gid", type=int,
                   help="optional dedicated group permitted to connect (0660)")
    p = sub.add_parser("evaluator-keygen", help="create evaluator Ed25519 keypair for separate trust domains")
    p.add_argument("--private", required=True)
    p.add_argument("--public", required=True)
    p = sub.add_parser("witness-keygen", help="create separate v5 witness Ed25519 keypair")
    p.add_argument("--private", required=True)
    p.add_argument("--public", required=True)
    p = sub.add_parser("prereg-witness", help="sign v5 preregistration at independent witness")
    p.add_argument("--prereg", required=True)
    p.add_argument("--witness-private-key", required=True)
    p.add_argument("--witness-public-key", required=True)
    p.add_argument("--witness-identity", default="independent_witness")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prereg-create", help="generate PRIVATE v5 witnessed or v4 config")
    p.add_argument("--evaluator-public-key", help="evaluator public key to freeze in prereg")
    p.add_argument("--witness-public-key", help="independently managed witness public key for v5")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--episodes", type=int, default=100)
    p = sub.add_parser("ledger-demo", help="record synthetic evidence")
    p.add_argument("--db", default="artifacts/evidence.sqlite")
    p.add_argument("--anchor")
    p = sub.add_parser("ledger-verify", help="check chain against optional external hash anchor")
    p.add_argument("--db", required=True)
    p.add_argument("--anchor")
    args = parser.parse_args(argv)
    try:
        commands = {"demo": cmd_demo, "session-demo": cmd_session,
                    "train": cmd_train, "evaluate": cmd_evaluate,
                    "qualify": cmd_qualify, "candidate-authorize": cmd_candidate_authorize,
                    "candidate-stage": cmd_candidate_stage,
                    "init-key": cmd_init_key,
                    "gate-server": cmd_gate, "prereg-create": cmd_prereg_create,
                    "evaluator-keygen": cmd_evaluator_keygen,
                    "witness-keygen": cmd_witness_keygen,
                    "prereg-witness": cmd_prereg_witness,
                    "ledger-demo": cmd_ledger_demo,
                    "ledger-verify": cmd_ledger_verify}
        result = commands[args.cmd](args)
        return int(result or 0)
    except (ValueError, FileNotFoundError, PermissionError, KeyError) as exc:
        print(f"SAAL error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
