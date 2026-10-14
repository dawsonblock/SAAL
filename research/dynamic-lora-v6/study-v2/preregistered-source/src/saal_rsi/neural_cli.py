"""Explicit v0.9.3 commands; training cannot qualify or activate a candidate."""
import argparse,json,sys
from pathlib import Path
from .cli import _read,_write
from .assurance import load_signing_key
from .model_contract import sign,verify
from .neural_contract import ModelSpec,new_transformer_job
from .research_datasets import semantic_benchmark
from .neural_backend import baseline_bytes
from .neural_campaign import campaign_protocol,execute_campaign,campaign_summary

def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('neural-data');c.add_argument('--out',required=True);c.add_argument('--train-pairs',type=int,default=16);c.add_argument('--diagnostic-pairs',type=int,default=4)
    c=sub.add_parser('neural-job-create');c.add_argument('--dataset',required=True);c.add_argument('--controller-key',required=True);c.add_argument('--out',required=True);c.add_argument('--baseline');c.add_argument('--mode',choices=('frozen','lora','full'),default='lora');c.add_argument('--rank',type=int,choices=(2,4,8),default=4);c.add_argument('--seed',type=int,default=17);c.add_argument('--baseline-steps',type=int,default=300);c.add_argument('--adaptation-steps',type=int,default=500);c.add_argument('--max-seconds',type=int,default=600)
    c=sub.add_parser('neural-campaign');c.add_argument('--dataset',required=True);c.add_argument('--controller-key',required=True);c.add_argument('--workspace',required=True);c.add_argument('--backend',choices=('local','colab'),default='local');c.add_argument('--seeds',type=int,nargs='+',default=[17,29,43,71,101]);c.add_argument('--baseline-steps',type=int,default=300);c.add_argument('--adaptation-steps',type=int,default=500);c.add_argument('--max-run-seconds',type=int,default=600)
    c=sub.add_parser('neural-report');c.add_argument('--outcomes',required=True);c.add_argument('--controller-public-key',required=True);c.add_argument('--out',required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='neural-data':result=semantic_benchmark(a.train_pairs,a.diagnostic_pairs);_write(a.out,result);result={'dataset':a.out,'records':len(result['records']),'qualification_status':'DIAGNOSTIC_ONLY'}
        elif a.command=='neural-report':
            result=campaign_summary(_read(a.outcomes)['runs'],a.controller_public_key);_write(a.out,result);result={'report':a.out,'qualification_status':'NOT_EVALUATED'}
        else:
            key=load_signing_key(Path(a.controller_key));pin=key.public_key().public_bytes_raw().hex();dataset=_read(a.dataset)
            if a.command=='neural-job-create':
                import hashlib
                spec=ModelSpec(initialization_seed=a.seed,adapter_configuration={'rank':a.rank,'targets':['query','value'],'scale':1.0}).as_dict()
                # Hash and bound the external immutable predecessor; tensor inventory is checked on execution.
                if a.baseline:
                    from .spec import sha
                    raw=Path(a.baseline).read_bytes();digest=hashlib.sha256(raw).hexdigest();baseline_bytes(a.baseline,digest)
                else:digest=None
                job=new_transformer_job(dataset,model_spec=spec,seed=a.seed,mode=a.mode,baseline_steps=a.baseline_steps,adaptation_steps=a.adaptation_steps,baseline_sha256=digest,max_seconds=a.max_seconds)
                _write(a.out,sign(job,key));result={'job':a.out,'controller_public_key':pin}
            else:
                protocol=campaign_protocol(dataset,seeds=a.seeds,baseline_steps=a.baseline_steps,adaptation_steps=a.adaptation_steps,max_run_seconds=a.max_run_seconds,backend=a.backend)
                report=execute_campaign(sign(protocol,key),pin,a.controller_key,dataset,a.workspace)
                result={'report':str(Path(a.workspace).resolve()/'report.json'),'completed_runs':len(report['runs']),'qualification_status':report['qualification_status']}
        print(json.dumps(result,allow_nan=False));return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:print(json.dumps({'error':type(exc).__name__,'detail':str(exc)}),file=sys.stderr);return 2
