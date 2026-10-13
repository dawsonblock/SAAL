"""Fresh CPU reconstruction of completed v4 models, with bounded waiting."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/reliable-learning-v3/tools'))
from saal_rsi.model_contract import verify
from saal_rsi.epoch_artifacts import validate_bundle
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.spec import sha,canonical
from reproduce_cpu import verified_measurement


def replay(bundle,pin,cache,out):
    job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
    native=validate_bundle(bundle,job,pin);spec=job['model_spec']
    snapshot=Path(cache)/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
    import torch
    torch.set_num_threads(4)
    backend=create_backend(spec,snapshot,job['seed']).load_pretrained()
    if next(backend.model.parameters()).device.type!='cpu':raise ValueError('CPU reconstruction required')
    start=time.monotonic();measurements=[]
    for name in ('initial','best','terminal'):
        if time.monotonic()-start>2400:raise TimeoutError('bounded replay')
        backend.restore_checkpoint(bundle/(name+'.safetensors'))
        for split in ('development','regression'):
            rows=[r for r in job['dataset']['records'] if r['split']==split]
            restored=backend.evaluate(rows);original=native['snapshots'][name]['predictions'][split]
            agreement=prediction_equal(restored,original,spec['model_kind'],1e-4)
            delta=max(abs(a-b) for p,q in zip(restored,original) for a,b in zip(p['probabilities'],q['probabilities']))
            measurements.append({'checkpoint':name,'split':split,'records':len(rows),
                'prediction_agreement':agreement,'max_probability_difference':delta,'predictions':restored})
            print(json.dumps({'run':bundle.name,'checkpoint':name,'split':split,'agreement':agreement,'max_probability_difference':delta}),flush=True)
    record={'schema':'saal-public-controller-cpu-replay-v1','run':bundle.name,'job_digest':sha(job),
        'runtime_dependencies':verify_dependencies(),'device':'cpu','probability_tolerance':1e-4,
        'all_predictions_agree':all(m['prediction_agreement'] for m in measurements),
        'measurements':measurements,'frozen_integrity':backend.verify_frozen_weights(),
        'elapsed_seconds':time.monotonic()-start,'independent_evaluator':False,
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    verified_measurement(record,native,sha(job))
    with out.open('xb') as f:f.write(canonical(record))


def watch(cache):
    output=PROJECT/'cpu-replay';output.mkdir(exist_ok=True);deadline=time.monotonic()+18000
    while True:
        count=0
        for group,folder,artifacts,pin_file in [
            ('head',PROJECT/'legacy-head/colab',PROJECT/'legacy-head/artifacts',ROOT/'research/reliable-learning-v3/head-control/experiment/controller.pub'),
            ('paired',PROJECT/'experiment/colab-paired',PROJECT/'experiment/artifacts',PROJECT/'experiment/controller.pub'),
            ('continuation',PROJECT/'experiment/continuation-01/colab',PROJECT/'experiment/artifacts',PROJECT/'experiment/controller.pub'),
            ('full',PROJECT/'experiment/colab-full',PROJECT/'experiment/artifacts',PROJECT/'experiment/controller.pub')]:
            path=folder/'outcomes.json'
            if not path.exists():continue
            for run in json.loads(path.read_text())['runs']:
                if run['state']!='COMPLETED':continue
                count+=1;destination=output/(group+'-'+run['run']+'.json')
                if destination.exists():continue
                replay(artifacts/run['run'],pin_file.read_text().strip(),cache,destination)
        full_state=PROJECT/'experiment/colab-full/monitor-state.json'
        rejected=PROJECT/'experiment/full-storage-admission-rejected.json'
        if full_state.exists() and json.loads(full_state.read_text())['terminal'] or rejected.exists():
            print(json.dumps({'completed_replay_records':len(list(output.glob('*.json'))),'completed_native_runs_seen':count}),flush=True)
            return
        if time.monotonic()>deadline:raise TimeoutError('bounded replay watcher')
        print(json.dumps({'waiting_for_native_runs':True,'completed_replay_records':len(list(output.glob('*.json')))}),flush=True)
        time.sleep(30)


if __name__=='__main__':watch(sys.argv[1])
