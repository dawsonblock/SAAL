"""Research-only interface. Keep trainer/qualifier on separate hosts in non-demo deployments."""
from __future__ import annotations
import argparse
import json
import secrets
import sys
import sqlite3
from pathlib import Path
from .spec import canonical,sha,parse_unique_json
from .discover import discover
from .assurance import (create_private_plan,validate_proposal,qualify,create_key,
                        load_signing_key,sign_qualification,verify_attestation)


from contextlib import contextmanager
import os
import stat

@contextmanager
def _output(path):
    """Reserve output before spending an anchor; clean only our failed partial file."""
    out=Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    owned=os.fstat(fd);written=False
    try:
        with os.fdopen(fd,'wb') as stream:
            def save(obj):
                nonlocal written
                if written:raise RuntimeError('output already written')
                data=canonical(obj)+b'\n'
                stream.write(data);stream.flush();os.fsync(stream.fileno());written=True
            yield save
            if not written:raise RuntimeError('output not produced')
    except BaseException:
        try:
            current=out.lstat()
            if (current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino):out.unlink()
        except FileNotFoundError:pass
        raise


def _write(path,obj):
    canonical(obj)  # reject unserializable data before reserving a path
    with _output(path) as save:save(obj)


def _read(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,'O_NONBLOCK',0))
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):raise ValueError('input must be regular file')
        if info.st_size>2_000_000:raise ValueError('input too large')
        with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(2_000_001)
        if len(data)>2_000_000:raise ValueError('input too large')
        return parse_unique_json(data)
    finally:os.close(fd)


def args_seeds(s:str)->list[int]:
    seeds=[int(v.strip()) for v in s.split(',') if v.strip()]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('unique seeds required')
    return seeds


def main(argv=None)->int:
    p=argparse.ArgumentParser(prog='saal-rsi',description='SAAL + Dream-RSI governed synthetic research')
    sub=p.add_subparsers(required=True,dest='cmd')
    d=sub.add_parser('discover',help='development only: replay proposal, no holdout')
    d.add_argument('--out',required=True);d.add_argument('--development-seeds',default='111,112,113')
    d.add_argument('--selection-seeds',default='211,212,213');d.add_argument('--worlds',type=int,default=3)
    d.add_argument('--episodes',type=int,default=8)
    k=sub.add_parser('keygen',help='evaluator-only: create Ed25519 signing key')
    k.add_argument('--out',required=True)
    f=sub.add_parser('plan-create',help='evaluator-only: prepare private holdout plan')
    f.add_argument('--out',required=True);f.add_argument('--seeds');f.add_argument('--episodes',type=int,default=100)
    f.add_argument('--min-gain',type=float,default=0.)
    q=sub.add_parser('qualify',help='evaluator-only: sealed independent tests, signed result')
    q.add_argument('--proposal',required=True);q.add_argument('--private-plan',required=True)
    q.add_argument('--signing-key',required=True);q.add_argument('--out',required=True)
    q.add_argument('--receipt',required=True);q.add_argument('--registry',required=True)
    q.add_argument('--anchor-public-key',required=True)
    anchor=sub.add_parser('anchor',help='custodian-only: sign and reserve preregistration before evaluation')
    anchor.add_argument('--proposal',required=True);anchor.add_argument('--private-plan',required=True)
    anchor.add_argument('--custodian-key',required=True);anchor.add_argument('--registry',required=True)
    anchor.add_argument('--out',required=True)
    v=sub.add_parser('verify',help='verify signed result with pinned evaluator public key')
    v.add_argument('--report',required=True);v.add_argument('--public-key',required=True)
    v.add_argument('--proposal');v.add_argument('--anchor-public-key',required=True)
    a=p.parse_args(argv)
    try:
        if a.cmd=='discover':
            out=discover(development_seeds=args_seeds(a.development_seeds),
                selection_seeds=args_seeds(a.selection_seeds),worlds=a.worlds,episodes=a.episodes)
            _write(a.out,out)
            print(json.dumps({'proposal':a.out,'candidate':out['candidate']['name'],
                              'qualification_status':'NOT_EVALUATED'},sort_keys=True))
        elif a.cmd=='keygen':
            public=create_key(Path(a.out))
            print(json.dumps({'key_path':a.out,'public_key_hex':public}))
        elif a.cmd=='plan-create':
            if a.seeds:seeds=args_seeds(a.seeds)
            else:
                s=set()
                while len(s)<6:s.add(secrets.randbelow(2**30))
                seeds=sorted(s)
            plan=create_private_plan(seeds=seeds,cases_per_seed=a.episodes,min_improvement=float(a.min_gain))
            _write(a.out,plan)
            print(json.dumps({'plan_path':a.out,'private_plan_commitment':sha(plan),
                              'note':'Run custodian anchor before evaluation. Protect and externally publish signed receipt.'}))
        elif a.cmd=='anchor':
            from .registry import Registry,public_hex
            key=load_signing_key(Path(a.custodian_key))
            proposal=_read(a.proposal);plan=_read(a.private_plan)
            with _output(a.out) as save:
                with Registry(a.registry,public_hex(key)) as registry:
                    receipt=registry.anchor(proposal,plan,key)
                save(receipt)
            print(json.dumps({'receipt':a.out,'anchor_digest':sha(receipt)}))
        elif a.cmd=='qualify':
            proposal=_read(a.proposal);plan=_read(a.private_plan)
            from .registry import Registry
            key=load_signing_key(Path(a.signing_key))
            from .registry import public_hex
            from .validation import key_bytes
            if key_bytes(public_hex(key))==key_bytes(a.anchor_public_key):raise ValueError('distinct custodian and evaluator keys required')
            receipt=_read(a.receipt)
            with _output(a.out) as save:
                with Registry(a.registry,a.anchor_public_key) as registry:
                    report=qualify(proposal,plan,registry=registry,receipt=receipt)
                att=sign_qualification(report,key)
                save(att)
            print(json.dumps({'report':a.out,'qualification_passed':report['qualification_passed'],
                              'promotion_allowed':False,'paired_gain_lower':report['paired_gain_lower_alpha']}))
        elif a.cmd=='verify':
            report=_read(a.report)
            proposal=_read(a.proposal) if a.proposal else None
            good=verify_attestation(report,a.public_key,proposal=proposal,anchor_public_key=a.anchor_public_key)
            print(json.dumps({'verified':good,'qualification_passed':report.get('report',{}).get('qualification_passed') if good else None,
                              'promotion_allowed':False}))
            return 0 if good else 2
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,sqlite3.Error) as exc:
        print(json.dumps({'error':type(exc).__name__,'detail':str(exc)}),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
