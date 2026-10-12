"""Role-host CLI. Keys and protected plans stay on independently managed hosts."""
import argparse
import os
from pathlib import Path
from .spec import parse_unique_json, canonical
from .model_contract import verify, sign
from .attempt_service import private_key, TLSClient
from .pretrained_qualification import freeze, make_protocol, validate_protocol, bindings, reserve, evaluate, verify_report


def main():
    parser = argparse.ArgumentParser(prog='saal-pretrained-qualification')
    sub = parser.add_subparsers(dest='command', required=True)
    frozen = sub.add_parser('freeze'); frozen.add_argument('--bundle', required=True)
    frozen.add_argument('--controller-pin', required=True); frozen.add_argument('--snapshot', choices=('initial','best','terminal'), default='terminal')
    replay = sub.add_parser('replay-public')
    plan = sub.add_parser('custodian-plan')
    for command in (replay, plan):
        command.add_argument('--candidate-bundle', required=True); command.add_argument('--baseline-bundle', required=True)
        command.add_argument('--candidate-snapshot', choices=('initial','best','terminal'), default='terminal')
        command.add_argument('--baseline-snapshot', choices=('initial','best','terminal'), default='initial')
        command.add_argument('--controller-pin', required=True); command.add_argument('--private-key', required=True)
    replay.add_argument('--cache', required=True)
    plan.add_argument('--holdout', required=True); plan.add_argument('--custody-review', required=True)
    plan.add_argument('--replay-review', required=True); plan.add_argument('--pins', required=True)
    plan.add_argument('--question-digest', required=True); plan.add_argument('--selection-digest', required=True)
    reserve = sub.add_parser('custodian-reserve'); reserve.add_argument('--mirror', required=True)
    reserve.add_argument('--enrollment', required=True); reserve.add_argument('--witness-client', required=True)
    run = sub.add_parser('evaluate'); run.add_argument('--candidate-bundle', required=True)
    run.add_argument('--baseline-bundle', required=True); run.add_argument('--reservation', required=True)
    run.add_argument('--witness-client', required=True); run.add_argument('--enrollment', required=True)
    run.add_argument('--cache', required=True); run.add_argument('--private-key', required=True)
    check = sub.add_parser('verify'); check.add_argument('--report', required=True)
    for command in (reserve, run, check):
        command.add_argument('--protocol', required=True); command.add_argument('--custodian-pin', required=True)
    for command in (frozen, replay, plan, reserve, run, check):
        command.add_argument('--out', required=True)
    args = parser.parse_args()
    def read(path):
        p = Path(path)
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 50_000_000:
            raise ValueError('bounded regular input file required')
        return parse_unique_json(p.read_bytes())
    # Reserve a fresh output before any authority transition. Never overwrite
    # evidence or run an experiment whose result cannot be retained.
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('fresh evidence output required')
    descriptor = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as destination:
        try:
            value = execute(args, read)
            destination.write(canonical(value))
        except BaseException as exc:
            destination.write(canonical({'schema':'saal-qualification-operation-incident-v1',
                'operation':args.command,'error_type':type(exc).__name__,'status':'FAILED',
                'attempt_state':'CONSULT_WITNESS_NO_IMPLICIT_REFUND',
                'external_deployment_allowed':False}))
            raise
        finally:
            destination.flush();os.fsync(destination.fileno())
    print('Evidence saved. External deployment remains prohibited.')


def execute(args, read):
    protocol = None
    if args.command in ('custodian-reserve','evaluate','verify'):
        protocol = verify(read(args.protocol), args.custodian_pin)
        validate_protocol(protocol)
        if protocol['pins']['custodian'] != args.custodian_pin:
            raise PermissionError('custodian protocol signer substitution')
    if args.command == 'freeze':
        value = freeze(args.bundle, args.controller_pin, args.snapshot)
    elif args.command in ('replay-public','custodian-plan'):
        candidate = freeze(args.candidate_bundle, args.controller_pin, args.candidate_snapshot)
        baseline = freeze(args.baseline_bundle, args.controller_pin, args.baseline_snapshot)
        key = private_key(args.private_key)
        if args.command == 'replay-public':
            from .pretrained_inference import replay_public
            value = sign(replay_public(args.candidate_bundle, args.baseline_bundle, candidate, baseline, args.cache, args.controller_pin), key)
        else:
            job = verify(read(Path(args.candidate_bundle)/'signed_training_job.json'), args.controller_pin)
            pins = read(args.pins)
            if key.public_key().public_bytes_raw().hex() != pins['custodian']:
                raise PermissionError('custodian signing identity')
            value = sign(make_protocol(candidate, baseline, job['dataset'], read(args.holdout), read(args.custody_review), pins,
                question_digest=args.question_digest, selection_digest=args.selection_digest, replay_review=read(args.replay_review)), key)
    elif args.command == 'custodian-reserve':
        from .attempt_authority import CustodianMirror
        client = TLSClient(read(args.witness_client))
        mirror = CustodianMirror(args.mirror, client, read(args.enrollment), protocol['pins']['witness'])
        value = reserve(protocol, mirror, client, read(args.enrollment))
    elif args.command == 'evaluate':
        key = private_key(args.private_key)
        if key.public_key().public_bytes_raw().hex() != protocol['pins']['evaluator']:
            raise PermissionError('evaluator signing identity')
        value = sign(evaluate(protocol, args.candidate_bundle, args.baseline_bundle,
            read(args.reservation), TLSClient(read(args.witness_client)), read(args.enrollment), args.cache), key)
    else:
        value = verify_report(read(args.report), protocol)
    return value


if __name__ == '__main__':
    main()
