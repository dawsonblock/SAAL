"""Matched, preregistered public research campaigns. Never protected qualification."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys
from .spec import canonical,sha,parse_unique_json
from .neural_contract import ModelSpec,new_transformer_job,worker_digest,SELECTION,rows
from .model_contract import sign,verify
from .model_backend import JobBudget,bundle_digest
from .neural_artifacts import validate_neural_bundle
from .neural_training import probability_metrics,selection_score
from .experiment_registry import ExperimentRegistry
from .neural_backend import run_neural_local,neural_colab_plan,attach_cleanup,CLOUD_OVERHEAD


def campaign_protocol(dataset,*,seeds=(17,29,43,71,101),baseline_steps=300,adaptation_steps=500,max_run_seconds=600,backend='local'):
    spec=ModelSpec().as_dict()
    # Complete admission occurs before registry reservation.
    new_transformer_job(dataset,model_spec=spec,mode='frozen',baseline_steps=baseline_steps,adaptation_steps=0,max_seconds=max_run_seconds)
    return {'schema':'saal-neural-campaign-v1','seeds':list(seeds),'modes':['frozen','full','lora-r4','lora-r8'],
            'baseline_steps':baseline_steps,'adaptation_steps':adaptation_steps,'dataset_digest':sha(dataset),
            'worker_digest':worker_digest(),'model_spec':spec,'max_run_seconds':max_run_seconds,'backend':backend,
            'selection_policy':SELECTION,'controller_digest':__import__('saal_rsi.identity',fromlist=['fingerprint']).fingerprint(),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}


def select_neural(candidates,pin):
    scores=[];development_digest=None
    for envelope,bundle in candidates:
        job=verify(envelope,pin);result=validate_neural_bundle(bundle,job,replay=True,controller_pin=pin)
        dev=rows(job['dataset'],'development');reg=rows(job['dataset'],'regression')
        if development_digest not in (None,sha(dev)):raise ValueError('same development population required')
        development_digest=sha(dev)
        d=probability_metrics(dev,result['development_predictions']['candidate']['probabilities'])
        r=probability_metrics(reg,result['regression_predictions']['candidate']['probabilities'])
        b=probability_metrics(reg,result['regression_predictions']['baseline']['probabilities'])
        accounting=parse_unique_json((Path(bundle)/'resource_accounting.json').read_bytes())
        scores.append({'job_digest':sha(job),'artifact_digest':bundle_digest(bundle,job),'score':list(selection_score(d,r,b)),
                       'development':d,'resource_seconds':accounting['elapsed_seconds'],'trainable_parameters':accounting['trainable_parameters']})
    if not scores or len(scores)>20:raise ValueError('bounded candidate inventory')
    scores.sort(key=lambda r:(*[-x for x in r['score']],r['trainable_parameters'],r['resource_seconds'],r['artifact_digest']))
    return {'schema':'saal-neural-selection-v1','ranking':scores,'selected_artifact':scores[0]['artifact_digest'],'selected_job_digest':scores[0]['job_digest'],
            'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'uses_final_holdout':False}


def run_registered_jobs(protocol_envelope,pin,key,dataset,root,*,remote_session=None):
    protocol=verify(protocol_envelope,pin)
    from .identity import fingerprint
    if protocol['controller_digest']!=fingerprint():raise ValueError('campaign controller source drift')
    if protocol['worker_digest']!=worker_digest() or protocol['dataset_digest']!=sha(dataset) or protocol['selection_policy']!=SELECTION:raise ValueError('campaign source/data/policy drift')
    root=Path(root);results=[]
    with JobBudget(root/'submission-budget.db',pin,max_jobs=len(protocol['seeds'])*4,max_seconds=len(protocol['seeds'])*4*(protocol['max_run_seconds']+30)) as ledger:
        try:
            for seed in protocol['seeds']:
                baseline=None
                for mode in protocol['modes']:
                    spec=copy.deepcopy(protocol['model_spec']);spec['initialization_seed']=seed
                    spec['adapter_configuration']['rank']=8 if mode=='lora-r8' else 4
                    job=new_transformer_job(dataset,model_spec=spec,seed=seed,mode='frozen' if mode=='frozen' else 'full' if mode=='full' else 'lora',
                        baseline_steps=protocol['baseline_steps'] if baseline is None else 0,
                        adaptation_steps=0 if mode=='frozen' else protocol['adaptation_steps'],
                        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline else None,max_seconds=protocol['max_run_seconds'])
                    envelope=sign(job,key);attempt=root/f'seed-{seed}'/mode;attempt.mkdir(parents=True,exist_ok=False)
                    (attempt/'job.json').write_bytes(canonical(envelope));ledger.reserve(envelope,job['max_seconds']+30)
                    print(json.dumps({'monitor':'campaign_progress','seed':seed,'mode':mode,'completed_runs':len(results),'total_runs':len(protocol['seeds'])*4}),flush=True)
                    out=attempt/'bundle'
                    try:
                        if remote_session is None:run_neural_local(envelope,pin,out,baseline)
                        else:
                            from .model_backend import unpack_result
                            from .neural_worker import FILES
                            from .model_monitor import stream_command
                            plan=neural_colab_plan(envelope,pin,attempt/'remote',baseline)
                            runner=attempt/'remote/run.py';runner.write_text(runner.read_text().replace('/content/saal_result','/content/saal_result_'+sha(job).split(':')[1][:12]))
                            for i,(command,phase) in enumerate(zip(plan['commands'][:-1],plan['phases'])):
                                if phase not in ('UPLOAD_JOB','UPLOAD_BASELINE','TRAINING','DOWNLOAD'):continue
                                command=[remote_session if x==plan['session'] else x for x in command]
                                if phase=='DOWNLOAD':command=[x.replace('/content/saal_result.zip','/content/saal_result_'+sha(job).split(':')[1][:12]+'.zip') for x in command]
                                stream_command(command,attempt/(str(i)+'-'+phase.lower()+'.log'),job['max_seconds']+30,
                                    lambda r:print(json.dumps(r,allow_nan=False),flush=True))
                            unpack_result(attempt/'remote/result.zip',out,names=FILES);validate_neural_bundle(out,job,replay=True)
                            runtime=parse_unique_json((out/'runtime_metrics.json').read_bytes())
                            if runtime['device']!='cuda' or job['gpu'] not in runtime['gpu']:raise ValueError('campaign GPU identity')
                        result=validate_neural_bundle(out,job,replay=True)
                        if baseline is None:baseline=out/'baseline.safetensors'
                        elif baseline.read_bytes()!=(out/'baseline.safetensors').read_bytes():raise ValueError('matched baseline bytes changed')
                        results.append({'seed':seed,'mode':mode,'status':'COMPLETED','job_digest':sha(job),'bundle':str(out.resolve()),'job':str((attempt/'job.json').resolve())})
                    except BaseException as exc:
                        results.append({'seed':seed,'mode':mode,'status':'INTERRUPTED','job_digest':sha(job),'error':{'type':type(exc).__name__,'detail':str(exc)}});raise
        finally:(root/'run-outcomes.json').write_bytes(canonical({'runs':results}))
    return results


def campaign_summary(results,pin):
    completed=[r for r in results if r['status']=='COMPLETED'];candidates=[];metrics=[]
    from .benchmarks import lexical_shortcut
    from collections import Counter
    for run in completed:
        envelope=parse_unique_json(Path(run['job']).read_bytes());job=verify(envelope,pin);result=validate_neural_bundle(run['bundle'],job,replay=True,controller_pin=pin)
        dev=rows(job['dataset'],'development');reg=rows(job['dataset'],'regression')
        candidate=probability_metrics(dev,result['development_predictions']['candidate']['probabilities'])
        baseline=probability_metrics(dev,result['development_predictions']['baseline']['probabilities'])
        # Shortcut comparison is declared diagnostic; not optimized on final outcomes.
        lexical=[lexical_shortcut(r['prompt']) for r in dev]
        majority=Counter(r['expected_decision'] for r in rows(job['dataset'],'train')).most_common(1)[0][0]
        controls={'lexical_accuracy':sum(p==r['expected_decision'] for p,r in zip(lexical,dev))/len(dev),'majority_accuracy':sum(r['expected_decision']==majority for r in dev)/len(dev)}
        metrics.append({'seed':run['seed'],'mode':run['mode'],'artifact_digest':bundle_digest(run['bundle'],job),'baseline_sha256':hashlib.sha256((Path(run['bundle'])/'baseline.safetensors').read_bytes()).hexdigest(),
            'development':candidate,'baseline_development':baseline,'regression':probability_metrics(reg,result['regression_predictions']['candidate']['probabilities']),
            'controls':controls,'resource_accounting':parse_unique_json((Path(run['bundle'])/'resource_accounting.json').read_bytes())})
        candidates.append((envelope,run['bundle']))
    distributions={mode:{'accuracy_values':[m['development']['accuracy'] for m in metrics if m['mode']==mode],
                         'pair_accuracy_values':[m['development']['pair_accuracy'] for m in metrics if m['mode']==mode]} for mode in ['frozen','full','lora-r4','lora-r8']}
    return {'schema':'saal-neural-campaign-report-v1','runs':metrics,'distributions':distributions,
            'selection':select_neural(candidates,pin) if candidates else None,'interrupted':[r for r in results if r['status']!='COMPLETED'],
            'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ACHIEVED','external_deployment_allowed':False,
            'limits':'All measurements are disclosed public synthetic diagnostics. Seeds repeat training, not independent family sampling. Compute is counted including unsuccessful runs; no transfer or protected qualification is inferred.'}


def execute_campaign(envelope,pin,key_path,dataset,workspace):
    from .assurance import load_signing_key
    from .model_monitor import execute_colab_plan,write_json
    key=load_signing_key(Path(key_path));protocol=verify(envelope,pin)
    if key.public_key().public_bytes_raw().hex()!=pin:raise ValueError('campaign controller identity')
    root=Path(workspace).resolve();root.mkdir(exist_ok=False)
    (root/'protocol.json').write_bytes(canonical(envelope));(root/'dataset.json').write_bytes(canonical(dataset))
    with ExperimentRegistry(root/'experiments.db') as registry:
        digest=registry.register(protocol);registry.start(digest)
        try:
            if protocol['backend']=='local':results=run_registered_jobs(envelope,pin,key,dataset,root)
            else:
                first=new_transformer_job(dataset,mode='frozen',baseline_steps=protocol['baseline_steps'],adaptation_steps=0,max_seconds=protocol['max_run_seconds'])
                setup=neural_colab_plan(sign(first,key),pin,root/'setup');commands=[];phases=[]
                for cmd,phase in zip(setup['commands'][:-1],setup['phases']):
                    if phase in ('PROVISIONING','UPLOAD_GPU-LOCK','UPLOAD_WORKER','DEPENDENCIES'):commands.append(cmd);phases.append(phase)
                coordinator=Path(__file__).resolve().parents[2]/'tools/run_campaign_coordinator.py'
                commands.append([sys.executable,'-I',str(coordinator),'--workspace',str(root),'--pin',pin,'--controller-key',str(Path(key_path).resolve()),'--session',setup['session']]);phases.append('CAMPAIGN')
                commands +=[['colab','log','-s',setup['session'],'-o',str(root/'runtime-log.jsonl')],['colab','stop','-s',setup['session']]];phases.append('EXPORT_LOGS')
                plan={'schema':'saal-colab-campaign-plan-v1','session':setup['session'],'job_digest':digest,'commands':commands,'phases':phases,
                      'estimated_max_seconds':len(protocol['seeds'])*4*(protocol['max_run_seconds']+30)+CLOUD_OVERHEAD,'billing_limit_enforced':False,'includes_final_holdout':False}
                write_json(root/'submission-plan.json',plan);state=execute_colab_plan(plan,root)
                results=parse_unique_json((root/'run-outcomes.json').read_bytes())['runs']
                for r in results:
                    if r['status']=='COMPLETED':attach_cleanup(r['bundle'],{'authority':'controller','confirmed':True,'status':'NAMED_SESSION_ABSENCE_CONFIRMED',
                        'session':setup['session'],'confirmation_log_sha256':hashlib.sha256((root/'cleanup-status.log').read_bytes()).hexdigest()})
            report=campaign_summary(results,pin);(root/'report.json').write_bytes(canonical(report));registry.finish(digest,report,completed=True)
            return report
        except BaseException as exc:
            outcomes=parse_unique_json((root/'run-outcomes.json').read_bytes())['runs'] if (root/'run-outcomes.json').exists() else []
            registry.finish(digest,{'runs':outcomes,'error':{'type':type(exc).__name__,'detail':str(exc)}},completed=False)
            raise
