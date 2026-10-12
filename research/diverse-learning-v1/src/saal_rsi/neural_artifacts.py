"""Strict v3 inventory, tensor checks and independent checkpoint reconstruction."""
from pathlib import Path
import hashlib
import math
import torch
from .neural_contract import validate_neural_job,DECISIONS,rows
from .neural_worker import FILES
from .neural_training import load_models,predict,probability_metrics,selection_score
from .spec import canonical,sha,parse_unique_json

MAX_BYTES=25_000_000

def validate_prediction(p,n):
    if type(p)is not dict or set(p)!={'labels','probabilities','logits'} or any(type(p[k])is not list or len(p[k])!=n for k in p):raise ValueError('prediction inventory')
    for label,probs,logits in zip(p['labels'],p['probabilities'],p['logits']):
        if label not in DECISIONS or type(probs)is not list or len(probs)!=4 or type(logits)is not list or len(logits)!=4:raise ValueError('decision/logit shape')
        if any(type(v)not in (float,int) or not math.isfinite(v) for v in probs+logits) or any(not 0<=v<=1 for v in probs) or not math.isclose(sum(probs),1,abs_tol=1e-5):raise ValueError('decision numeric')
        tensor=torch.tensor(logits,dtype=torch.float64)
        if not torch.allclose(tensor.softmax(-1),torch.tensor(probs,dtype=torch.float64),atol=1e-5,rtol=1e-5) or label!=DECISIONS[int(tensor.argmax())]:raise ValueError('decision/logit mismatch')
    return p

def validate_neural_bundle(directory,job,*,replay=False,controller_pin=None):
    validate_neural_job(job);root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('artifact directory')
    files=list(root.iterdir())
    if {p.name for p in files}!=FILES or any(p.is_symlink() or not p.is_file() for p in files):raise ValueError('neural artifact inventory')
    if any(p.stat().st_size>MAX_BYTES for p in files) or sum(p.stat().st_size for p in files)>MAX_BYTES:raise ValueError('neural artifact budget')
    read=lambda n:parse_unique_json((root/n).read_bytes())
    inventory=read('artifact_inventory.json')
    expected={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(FILES-{'artifact_inventory.json'})}
    if inventory!={'schema':'saal-neural-artifacts-v3','files':expected}:raise ValueError('artifact inventory hash substitution')
    result=read('result.json')
    fields={'schema','job_digest','model_spec_digest','worker_sha256','runtime','development_predictions','regression_predictions','qualification_status','qualification_passed','external_deployment_allowed','model_activation_allowed','status','mode','backbone_unchanged','artifact_hashes'}
    if type(result)is not dict or set(result)!=fields or result['schema']!='saal-model-result-v3' or result['job_digest']!=sha(job) or result['model_spec_digest']!=sha(job['model_spec']) or result['worker_sha256']!=job['worker_sha256'] or result['mode']!=job['mode']:raise ValueError('result/model/code binding')
    if result['qualification_status']!='NOT_EVALUATED' or result['status']!='UNTRUSTED_DEVELOPMENT_RESULT' or any(result[k] is not False for k in ('qualification_passed','external_deployment_allowed','model_activation_allowed')):raise ValueError('worker self-approval forbidden')
    if result['artifact_hashes']!={k:v for k,v in expected.items() if k!='result.json'}:raise ValueError('result artifact substitution')
    from .model_contract import verify
    envelope=read('signed_training_job.json')
    if controller_pin is not None:verify(envelope,controller_pin)
    if canonical(envelope['payload'])!=canonical(job):raise ValueError('artifact job substitution')
    # The caller verifies the controller pin before admission; a worker copy never becomes a new trust root.
    if read('model_spec.json')!=job['model_spec'] or read('dataset_manifest.json')!=job['dataset']:raise ValueError('model/data manifest substitution')
    runtime=result['runtime']
    if read('runtime_metrics.json')!=runtime or {k:v.split('+')[0] for k,v in runtime['versions'].items()}!=job['versions'] or runtime['numpy']!=job['serialization_numpy_version'] or runtime['model_spec_digest']!=sha(job['model_spec']):raise ValueError('runtime identity')
    for key,split in [('development_predictions','development'),('regression_predictions','regression')]:
        if set(result[key])!={'baseline','candidate'}:raise ValueError('prediction branch inventory')
        for p in result[key].values():validate_prediction(p,len(rows(job['dataset'],split)))
    baseline,candidate=load_models(root,job['model_spec'],job['mode'])
    inventory_expected={k:{'shape':list(v.shape),'dtype':str(v.dtype),'adapter':k.endswith(('.query.A','.query.B','.value.A','.value.B'))} for k,v in candidate.state_dict().items()}
    if read('tensor_inventory.json')!=inventory_expected:raise ValueError('tensor manifest')
    unchanged=all(torch.equal(v,candidate.state_dict()[k]) for k,v in baseline.state_dict().items() if not inventory_expected[k]['adapter'])
    if type(result['backbone_unchanged'])is not bool or result['backbone_unchanged']!=unchanged:raise ValueError('backbone modification claim')
    if job['baseline_sha256'] is not None and expected['baseline.safetensors']!=job['baseline_sha256']:raise ValueError('frozen baseline bytes modified')
    accounting=read('resource_accounting.json')
    if type(accounting)is not dict or set(accounting)!={'steps','training_tokens','elapsed_seconds','trainable_parameters','total_parameters','gpu_peak_bytes','budget_seconds','budget_steps','financial_billing_cap'}:raise ValueError('resource inventory')
    if any(type(accounting[k])is not int or accounting[k]<0 for k in ('steps','training_tokens','trainable_parameters','total_parameters','gpu_peak_bytes','budget_seconds','budget_steps')) or accounting['financial_billing_cap'] is not False:raise ValueError('resource numeric')
    if accounting['steps']!=job['baseline_steps']+job['adaptation_steps'] or accounting['budget_steps']!=accounting['steps'] or accounting['budget_seconds']!=job['max_seconds'] or type(accounting['elapsed_seconds'])not in (float,int) or not math.isfinite(accounting['elapsed_seconds']) or not 0<=accounting['elapsed_seconds']<=job['max_seconds']+30:raise ValueError('resource budget evidence')
    if accounting['training_tokens']>accounting['steps']*job['batch_size']*(job['model_spec']['max_sequence_bytes']+1):raise ValueError('training token budget')
    numel=lambda state:sum(t.numel() for t in state.values())
    expected_trainable=sum(t.numel() for k,t in candidate.state_dict().items() if job['mode']=='lora' and inventory_expected[k]['adapter'] or job['mode']=='full' and not inventory_expected[k]['adapter'])
    if accounting['total_parameters']!=numel(candidate.state_dict()) or accounting['trainable_parameters']!=expected_trainable:raise ValueError('parameter accounting')
    selection=read('checkpoint_selection.json');dev=rows(job['dataset'],'development');reg=rows(job['dataset'],'regression')
    if selection['policy']!=job['checkpoint_selection'] or selection['uses_protected_holdout'] is not False or selection['trained_steps']!=job['adaptation_steps'] or type(selection['selected_step'])is not int or not 0<=selection['selected_step']<=job['adaptation_steps'] or (selection['selected_step']%25 and selection['selected_step']!=job['adaptation_steps']):raise ValueError('checkpoint selection commitment')
    if selection['development']!=probability_metrics(dev,result['development_predictions']['candidate']['probabilities']) or selection['public_regression']!=probability_metrics(reg,result['regression_predictions']['candidate']['probabilities']):raise ValueError('checkpoint score reconstruction')
    from collections import Counter
    counts=Counter(r['expected_decision'] for r in rows(job['dataset'],'train'))
    weights=[len(rows(job['dataset'],'train'))/(len(counts)*counts[k]) if counts[k] else 0 for k in DECISIONS]
    if len(selection['class_weights'])!=4 or any(not math.isclose(x,y,abs_tol=1e-6) for x,y in zip(selection['class_weights'],weights)):raise ValueError('class weight evidence')
    history=[]
    for line in (root/'training_metrics.jsonl').read_bytes().splitlines():history.append(parse_unique_json(line))
    expected_steps=[(phase,s) for phase,steps in [('baseline',job['baseline_steps']),('adaptation',job['adaptation_steps'])] if phase=='adaptation' or job['baseline_sha256'] is None for s in range(steps+1) if s%25==0 or s==steps]
    if len(history)!=len(expected_steps):raise ValueError('training progress inventory')
    for entry,(phase,step) in zip(history,expected_steps):
        if entry['monitor']!='training_progress' or entry['phase']!=phase or type(entry['step'])is not int or entry['step']!=step or entry['total_steps']!=accounting['steps']:raise ValueError('training progress identity')
        if any(type(entry[k])not in (int,float) or not math.isfinite(entry[k]) or entry[k]<0 for k in ('objective_loss','elapsed_seconds','training_tokens','gpu_allocated_bytes','gpu_reserved_bytes')):raise ValueError('training telemetry numeric')
    selected=next(e for e in history if e['phase']=='adaptation' and e['step']==selection['selected_step'])
    if selected['development']!=selection['development'] or selected['public_regression']!=selection['public_regression']:raise ValueError('selected checkpoint telemetry')
    adaptation=[e for e in history if e['phase']=='adaptation'];baseline_reg=probability_metrics(reg,result['regression_predictions']['baseline']['probabilities'])
    for e in adaptation:
        score=list(selection_score(e['development'],e['public_regression'],baseline_reg))
        if e['selection_score']!=score:raise ValueError('selection score identity')
    best=max(adaptation,key=lambda e:tuple(e['selection_score']))
    if best['step']!=selection['selected_step']:raise ValueError('checkpoint selection order')
    cleanup=read('cleanup_evidence.json')
    if type(cleanup)is not dict or cleanup.get('authority')!='controller' or type(cleanup.get('confirmed'))is not bool or cleanup.get('status') not in ('HOST_CONFIRMATION_REQUIRED','LOCAL_CHILD_EXITED','NAMED_SESSION_ABSENCE_CONFIRMED'):raise ValueError('cleanup evidence shape')
    if cleanup['confirmed']!=(cleanup['status']!='HOST_CONFIRMATION_REQUIRED'):raise ValueError('cleanup evidence contradiction')
    if replay:
        for model,key in [(baseline,'baseline'),(candidate,'candidate')]:
            for split,field in [('development','development_predictions'),('regression','regression_predictions')]:
                restored=predict(model,rows(job['dataset'],split))
                reported=result[field][key]
                if restored['labels']!=reported['labels'] or not torch.allclose(torch.tensor(restored['logits']),torch.tensor(reported['logits']),atol=1e-4,rtol=1e-4):raise ValueError('independent checkpoint reconstruction mismatch')
    return result
