"""Additive learning/review commands; legacy training and authority stay intact."""
import argparse
from pathlib import Path
from .spec import canonical, parse_unique_json
from .pretrained_contract import job_bytes


def review_key(path):
    import os
    import stat
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != 32 or info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError('review key must be an owned private regular 32-byte file')
        raw = os.read(fd, 33)
        if len(raw) != 32: raise ValueError('review key size changed')
        return Ed25519PrivateKey.from_private_bytes(raw)
    finally: os.close(fd)


def read(path):
    return parse_unique_json(job_bytes(path))


def write(path, value):
    with Path(path).open('xb') as f: f.write(canonical(value))


def main():
    p = argparse.ArgumentParser(prog='python -m saal_rsi.reliability_cli')
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('corpus'); c.add_argument('--diverse', required=True); c.add_argument('--control', required=True); c.add_argument('--out', required=True)
    a = commands.add_parser('audit'); a.add_argument('--bundle', required=True); a.add_argument('--pin', required=True); a.add_argument('--out', required=True)
    r = commands.add_parser('retention'); r.add_argument('--parent', required=True); r.add_argument('--parent-pin', required=True); r.add_argument('--candidate', required=True); r.add_argument('--candidate-pin', required=True); r.add_argument('--max-family-drop', type=float, default=.05); r.add_argument('--out', required=True)
    q = commands.add_parser('review-packet'); q.add_argument('--dataset', required=True); q.add_argument('--out', required=True)
    s = commands.add_parser('sign-review'); s.add_argument('--dataset', required=True); s.add_argument('--draft', required=True); s.add_argument('--private-key', required=True); s.add_argument('--role', choices=['reviewer', 'adjudicator'], default='reviewer'); s.add_argument('--out', required=True)
    v = commands.add_parser('assess-review'); v.add_argument('--dataset', required=True); v.add_argument('--policy', required=True); v.add_argument('--reviews', nargs=2, required=True); v.add_argument('--semantic-review', required=True); v.add_argument('--adjudication'); v.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'corpus':
        from .reliability_corpus import build
        from .learning_dataset import diagnostic_audit
        from .public_review import packet, html_packet
        out = Path(args.out); out.mkdir(exist_ok=False)
        dataset, audit = build(read(args.diverse), read(args.control))
        write(out / 'dataset.json', dataset); write(out / 'state-audit.json', audit)
        write(out / 'lexical-screen.json', diagnostic_audit(dataset)); write(out / 'blind-review-draft.json', packet(dataset))
        (out / 'blind-review.html').write_text(html_packet(dataset))
    elif args.command == 'audit':
        from .learning_audit import audit_bundle
        write(args.out, audit_bundle(args.bundle, args.pin))
    elif args.command == 'retention':
        from .learning_audit import compare_retention
        write(args.out, compare_retention(args.parent, args.parent_pin, args.candidate, args.candidate_pin, max_family_drop=args.max_family_drop))
    elif args.command == 'review-packet':
        from .public_review import packet, html_packet
        out = Path(args.out); out.mkdir(exist_ok=False); dataset = read(args.dataset)
        write(out / 'blind-review-draft.json', packet(dataset)); (out / 'blind-review.html').write_text(html_packet(dataset))
    elif args.command == 'sign-review':
        from .public_review import sign_draft
        write(args.out, sign_draft(read(args.dataset), read(args.draft), review_key(args.private_key), role=args.role))
    elif args.command == 'assess-review':
        from .public_review import assess
        write(args.out, assess(read(args.dataset), read(args.policy), [read(x) for x in args.reviews], read(args.semantic_review), read(args.adjudication) if args.adjudication else None))


if __name__ == '__main__': main()
