"""Retention measures with explicit pre-task probes; no unmeasured transfer claims."""
import math

def retention_metrics(matrix,initial):
    if type(matrix)is not list or not matrix or type(initial)is not list or len(initial)!=len(matrix):raise ValueError('retention matrix dimensions')
    n=len(matrix)
    if any(type(row)is not list or len(row)!=n for row in matrix) or any(type(x)not in (float,int) or not math.isfinite(x) or not 0<=x<=1 for row in [initial,*matrix] for x in row):raise ValueError('retention accuracy')
    forgetting=[max(matrix[t][k] for t in range(k,n))-matrix[-1][k] for k in range(n)]
    bwt=[matrix[-1][k]-matrix[k][k] for k in range(n-1)]
    fwt=[matrix[k-1][k]-initial[k] for k in range(1,n)]
    return {'accuracy_matrix':matrix,'initial_accuracy':initial,'forgetting_by_task':forgetting,
            'mean_forgetting':sum(forgetting)/n,'backward_transfer':sum(bwt)/len(bwt) if bwt else 0.0,
            'forward_transfer':sum(fwt)/len(fwt) if fwt else 0.0,
            'limits':'Forward-transfer probes use reserved public diagnostics. This metric alone does not show faster adaptation than a matched fresh learner.'}


def sequential_adaptation(baseline,spec,task_groups,*,steps=100,replay_ratio=0.0,seed=17,max_seconds=300):
    """Public-only LoRA sequence with matched fresh controls and bounded training.

    Each group is {training: records, diagnostics: records}. Splits and group
    provenance must already have been admitted; this function rejects holdouts.
    Future probes are diagnostics only and never used for checkpoint selection.
    """
    import copy,time
    import torch
    from safetensors.torch import load_file
    from .model_architectures import initialized,validate_tensors,configure_training,tokenize
    from .neural_contract import DECISIONS
    from .neural_training import predict
    if type(task_groups)is not list or not 2<=len(task_groups)<=4 or type(steps)is not int or not 1<=steps<=500 or type(seed)is not int or type(replay_ratio)is not float or replay_ratio not in (0.0,.25,.5) or type(max_seconds)is not int or not 10<=max_seconds<=1800:raise ValueError('sequential experiment budget')
    families=set();training_families=set();diagnostic_families=set()
    for group in task_groups:
        if type(group)is not dict or set(group)!={'training','diagnostics'} or any(not group[k] for k in group):raise ValueError('task group inventory')
        for key,records in group.items():
            if any(r.get('split') not in (('train',) if key=='training' else ('development','regression')) or r.get('expected_decision') not in DECISIONS for r in records):raise ValueError('protected/non-public data prohibited')
            owned={r['task_family_id'] for r in records}
            if families&owned:raise ValueError('task-family reuse across groups/partitions')
            families|=owned
    torch.set_num_threads(2);torch.manual_seed(seed);started=time.monotonic();state=load_file(str(baseline));validate_tensors(state,spec,'backbone')
    shared=initialized(spec);shared.load_state_dict(state,strict=False)
    def accuracy(model,records):
        labels=predict(model,records)['labels'];return sum(p==r['expected_decision'] for p,r in zip(labels,records))/len(records)
    initial=[accuracy(shared,g['diagnostics']) for g in task_groups];matrix=[];fresh=[];seen=[];work=[]
    generator=torch.Generator().manual_seed(seed)
    def adapt(model,records):
        parameters=configure_training(model,'lora');opt=torch.optim.AdamW(parameters,lr=.001)
        tokens,mask=tokenize([r['prompt'] for r in records],spec);labels=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records]);tokens_processed=0
        for _ in range(steps):
            if time.monotonic()-started>max_seconds:raise TimeoutError('sequential experiment deadline')
            indices=torch.randint(len(records),(16,),generator=generator);model.train();opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(tokens[indices],mask[indices]),labels[indices]);loss.backward();opt.step();tokens_processed+=int(mask[indices].sum())
        return tokens_processed
    for stage,group in enumerate(task_groups):
        previous=seen[:int(len(group['training'])*replay_ratio/(1-replay_ratio))] if replay_ratio else []
        tokens=adapt(shared,group['training']+previous);control=initialized(spec);control.load_state_dict(state,strict=False)
        control_tokens=adapt(control,group['training']);fresh.append(accuracy(control,group['diagnostics']))
        matrix.append([accuracy(shared,g['diagnostics']) for g in task_groups]);seen.extend(group['training'])
        work.append({'stage':stage,'steps_per_branch':steps,'shared_tokens':tokens,'fresh_tokens':control_tokens,'replay_examples':len(previous),'trainable_parameters':sum(p.numel() for p in shared.parameters() if p.requires_grad)})
    result=retention_metrics(matrix,initial)
    result.update(fresh_control_accuracy=fresh,matched_adaptation_gain=[matrix[k][k]-fresh[k] for k in range(len(matrix))],resource_accounting=work,
                  elapsed_seconds=time.monotonic()-started,qualification_status='DIAGNOSTIC_ONLY',external_deployment_allowed=False)
    return result
