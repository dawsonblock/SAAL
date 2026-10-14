"""Corrective research tools, with explicit public-only status and no activation."""
import argparse
from pathlib import Path
from .spec import parse_unique_json, canonical, sha


def main():
    parser=argparse.ArgumentParser(prog='saal-repair')
    sub=parser.add_subparsers(dest='command',required=True)
    data=sub.add_parser('public-dataset');data.add_argument('--out',required=True)
    screen=sub.add_parser('screen');screen.add_argument('--dataset',required=True);screen.add_argument('--out',required=True)
    select=sub.add_parser('research-selection');select.add_argument('--snapshots',required=True);select.add_argument('--out',required=True)
    budget=sub.add_parser('init-development-budget');budget.add_argument('--ledger',required=True);budget.add_argument('--pin',required=True);budget.add_argument('--max-jobs',type=int,required=True);budget.add_argument('--max-seconds',type=int,required=True)
    protocol=sub.add_parser('sequence-protocol');protocol.add_argument('--job',required=True);protocol.add_argument('--pin',required=True);protocol.add_argument('--private-key',required=True);protocol.add_argument('--learning-report',required=True);protocol.add_argument('--group',nargs='+',action='append',required=True);protocol.add_argument('--policy',choices=('none','reservoir-v1','stratified-family-v1'),default='stratified-family-v1');protocol.add_argument('--replay-ratio',type=float,default=.25);protocol.add_argument('--steps-per-stage',type=int,default=10);protocol.add_argument('--max-seconds',type=int,default=600);protocol.add_argument('--out',required=True)
    run=sub.add_parser('sequence-local');run.add_argument('--job',required=True);run.add_argument('--protocol',required=True);run.add_argument('--pin',required=True);run.add_argument('--learning-report',required=True);run.add_argument('--cache',required=True);run.add_argument('--out',required=True)
    run.add_argument('--ledger',required=True)
    replay=sub.add_parser('sequence-replay');replay.add_argument('--bundle',required=True);replay.add_argument('--pin',required=True);replay.add_argument('--cache',required=True);replay.add_argument('--out',required=True)
    args=parser.parse_args()
    from .pretrained_contract import job_bytes
    def read(path):return parse_unique_json(job_bytes(path))
    def write(value):
        with Path(args.out).open('xb') as f:f.write(canonical(value))
    if args.command=='init-development-budget':
        from .learning_experiments import LearningExperimentRegistry
        registry=LearningExperimentRegistry(args.ledger,args.pin,args.max_jobs,args.max_seconds);registry.close()
        print('Development budget initialized; controller-owned, not rollback-resistant or a billing cap')
    elif args.command=='public-dataset':
        from .public_mechanisms import public_mechanism_dataset
        write(public_mechanism_dataset())
    elif args.command=='screen':
        from .learning_dataset import validate_dataset
        from .lineage_review import provenance_screen
        write(provenance_screen(validate_dataset(read(args.dataset))['records']))
    elif args.command=='research-selection':
        from .research_selection import select_research_snapshots
        write(select_research_snapshots(read(args.snapshots)['snapshots']))
    elif args.command=='sequence-protocol':
        from .model_contract import verify,sign
        from .attempt_service import private_key
        from .sequence_v2 import identity,validate_protocol
        job=verify(read(args.job),args.pin);report=read(args.learning_report)
        value={'schema':'saal-pretrained-sequence-v2','worker_identity':identity(),'job_digest':sha(job),'groups':args.group,'method':'shared','replay_ratio':args.replay_ratio,'steps_per_stage':args.steps_per_stage,'max_seconds':args.max_seconds,'positive_learning_report_digest':sha(report),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'replay_policy':args.policy,'replay_seed':71,'reservoir_capacity':512,'dropout_control':'seeded-per-update-v1'}
        validate_protocol(value,job);envelope=sign(value,private_key(args.private_key));verify(envelope,args.pin);write(envelope)
    elif args.command=='sequence-local':
        from .model_contract import verify
        from .learning_backend import verify_dependencies
        from .pretrained_registry import validate_snapshot
        from .sequence_v2 import run_sequence_v2
        from .sequence_v2 import validate_protocol
        from .learning_experiments import LearningExperimentRegistry
        envelope=read(args.job);job=verify(envelope,args.pin);spec=job['model_spec']
        if spec['compute_backend']!='local':raise ValueError('local execution requires separately admitted local configuration')
        verify_dependencies();snapshot=Path(args.cache)/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
        protocol_envelope=read(args.protocol);validate_protocol(verify(protocol_envelope,args.pin),job)
        if Path(args.out).exists():raise FileExistsError('new experiment output required')
        registry=LearningExperimentRegistry(args.ledger,args.pin);attempt=registry.reserve(envelope);registry.start(attempt)
        try:
            import torch;torch.set_num_threads(2)
            result=run_sequence_v2(envelope,protocol_envelope,args.pin,snapshot,args.out,read(args.learning_report))
            registry.finish(attempt,{'protocol_digest':sha(protocol_envelope),'completed_updates':result['completed_updates'],'qualification_status':'NOT_EVALUATED'},True)
        except BaseException as exc:
            registry.finish(attempt,{'error_type':type(exc).__name__,'qualification_status':'NOT_EVALUATED'},False)
            raise
        finally:registry.close()
        print('Completed updates:',result['completed_updates'],'qualification:',result['qualification_status'])
    else:
        from .sequence_replay_v2 import replay
        import torch;torch.set_num_threads(2)
        result=replay(args.bundle,args.pin,args.cache);write(result)
        if not result['per_record_agreement']:raise SystemExit('checkpoint replay mismatch')
        print('Full per-record replay passed; research qualification remains NOT_EVALUATED')


if __name__=='__main__':main()
