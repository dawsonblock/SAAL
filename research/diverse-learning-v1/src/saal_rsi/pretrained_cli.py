"""Explicit pretrained commands; legacy entry points keep their versioned semantics."""
import argparse,json
from pathlib import Path
from .spec import canonical,parse_unique_json

def main():
    p=argparse.ArgumentParser(prog='saal-pretrained');sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('dataset');d.add_argument('--kind',choices=['semantic','generative'],required=True);d.add_argument('--out',required=True)
    m=sub.add_parser('model-spec');m.add_argument('--repository',required=True);m.add_argument('--method',choices=['head','full','lora','none'],default='lora');m.add_argument('--backend',choices=['local','colab'],default='colab');m.add_argument('--rank',type=int,default=8);m.add_argument('--precision',choices=['float16','float32']);m.add_argument('--max-input-tokens',type=int,default=256);m.add_argument('--out',required=True)
    f=sub.add_parser('fetch');f.add_argument('--repository',required=True);f.add_argument('--revision',required=True);f.add_argument('--cache',required=True)
    j=sub.add_parser('sign-job');j.add_argument('--spec',required=True);j.add_argument('--dataset',required=True);j.add_argument('--private-key',required=True);j.add_argument('--out',required=True);j.add_argument('--steps',type=int,default=50);j.add_argument('--seed',type=int,default=17);j.add_argument('--max-seconds',type=int,default=900);j.add_argument('--learning-rate',type=float,default=.0001)
    j.add_argument('--batch-size',type=int);j.add_argument('--accumulation',type=int,default=1);j.add_argument('--checkpoint-interval',type=int,default=25);j.add_argument('--max-new-tokens',type=int,default=80);j.add_argument('--class-balance',action='store_true');j.add_argument('--initial-loss-scale',type=int,default=256)
    r=sub.add_parser('local');r.add_argument('--job',required=True);r.add_argument('--pin',required=True);r.add_argument('--out',required=True);r.add_argument('--cache',required=True)
    c=sub.add_parser('colab');c.add_argument('--jobs',nargs='+',required=True);c.add_argument('--pin',required=True);c.add_argument('--workspace',required=True);c.add_argument('--artifacts',required=True);c.add_argument('--ledger',required=True);c.add_argument('--max-seconds',type=int,required=True)
    v=sub.add_parser('verify');v.add_argument('--bundle',required=True);v.add_argument('--pin',required=True)
    q=sub.add_parser('sequence-protocol');q.add_argument('--job',required=True);q.add_argument('--pin',required=True);q.add_argument('--private-key',required=True);q.add_argument('--learning-report',required=True);q.add_argument('--group',nargs='+',action='append',required=True);q.add_argument('--method',choices=['shared','isolated'],default='shared');q.add_argument('--replay-ratio',type=float,default=0.0);q.add_argument('--steps-per-stage',type=int,default=100);q.add_argument('--max-seconds',type=int,default=900);q.add_argument('--out',required=True)
    s=sub.add_parser('sequence');s.add_argument('--job',required=True);s.add_argument('--protocol',required=True);s.add_argument('--pin',required=True);s.add_argument('--learning-report',required=True);s.add_argument('--workspace',required=True);s.add_argument('--artifacts',required=True)
    a=p.parse_args()
    def read(path):
        from .pretrained_contract import job_bytes
        return parse_unique_json(job_bytes(path))
    def write(path,value):
        with Path(path).open('xb') as f:f.write(canonical(value))
    if a.command=='dataset':
        from .learning_dataset import semantic_pilot,generative_pilot
        write(a.out,semantic_pilot() if a.kind=='semantic' else generative_pilot())
    elif a.command=='model-spec':
        from .pretrained_contract import model_spec
        precision=a.precision or ('float32' if a.backend=='local' or a.repository.startswith('Qwen/') else 'float16')
        write(a.out,model_spec(a.repository,a.method,a.rank,a.backend,precision,max_tokens=a.max_input_tokens))
    elif a.command=='fetch':
        from .pretrained_registry import fetch_snapshot
        print(fetch_snapshot(a.cache,a.repository,a.revision,lambda v:print(json.dumps(v),flush=True)))
    elif a.command=='sign-job':
        from .pretrained_contract import new_job
        from .model_contract import sign
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        spec=read(a.spec);job=new_job(spec,read(a.dataset),steps=a.steps,seed=a.seed,max_seconds=a.max_seconds,learning_rate=a.learning_rate,batch_size=a.batch_size if a.batch_size is not None else (1 if spec['model_kind']=='qwen35-generative-v1' else 4),accumulation=a.accumulation,checkpoint_interval=a.checkpoint_interval,max_new_tokens=a.max_new_tokens,class_balance=a.class_balance,initial_loss_scale=a.initial_loss_scale)
        write(a.out,sign(job,Ed25519PrivateKey.from_private_bytes(Path(a.private_key).read_bytes())))
    elif a.command=='local':
        from .pretrained_backend import run_local
        print(json.dumps(run_local(read(a.job),a.pin,a.out,a.cache)['resource_accounting']))
    elif a.command=='colab':
        from .pretrained_backend import execute_campaign
        print(json.dumps(execute_campaign(a.workspace,[read(x) for x in a.jobs],a.pin,a.artifacts,a.ledger,a.max_seconds)))
    elif a.command=='verify':
        from .pretrained_artifacts import validate_bundle
        from .model_contract import verify
        envelope=read(Path(a.bundle)/'signed_training_job.json');job=verify(envelope,a.pin)
        print(json.dumps({'verified':True,'qualification_status':validate_bundle(a.bundle,job,a.pin)['qualification_status']}))
    elif a.command=='sequence-protocol':
        from .pretrained_continual import validate_sequence,sequence_worker_identity
        from .model_contract import sign,verify
        from .spec import sha
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        job=verify(read(a.job),a.pin);report=read(a.learning_report)
        if report.get('adaptation_gain_terminal',{}).get('mean_gain',0)<=0 or report.get('real_parameter_updates')is not True:raise ValueError('positive ordinary-learning evidence required')
        protocol={'schema':'saal-pretrained-sequence-v1','worker_identity':sequence_worker_identity(),'job_digest':sha(job),'groups':a.group,'method':a.method,'replay_ratio':a.replay_ratio,'steps_per_stage':a.steps_per_stage,'max_seconds':a.max_seconds,'positive_learning_report_digest':sha(report),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
        validate_sequence(protocol,job);envelope=sign(protocol,Ed25519PrivateKey.from_private_bytes(Path(a.private_key).read_bytes()));verify(envelope,a.pin);write(a.out,envelope)
    elif a.command=='sequence':
        from .pretrained_sequence_backend import execute_sequence
        result=execute_sequence(a.workspace,read(a.job),read(a.protocol),a.pin,read(a.learning_report),a.artifacts)
        print(json.dumps({'cleanup_verified':result['cleanup_verified'],'qualification_status':result['result']['qualification_status']}))

if __name__=='__main__':main()
