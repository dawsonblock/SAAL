"""PEFT AdaLoRA on the official pinned encoder; physical rank remains initial rank."""
from saal_rsi.modernbert_backend import ModernBERTBackend
from saal_rsi.pretrained_contract import model_spec
from .contract import ADAPTER
class DynamicBackend(ModernBERTBackend):
    def __init__(self,reference,snapshot):
        self.reference=reference
        spec=model_spec('answerdotai/ModernBERT-base',method='head',backend='local',precision='float32',max_tokens=256,budget=reference['model_spec']['resource_budget_id'])
        super().__init__(spec,snapshot,reference['seed'])
    def load_pretrained(self):
        import torch
        from peft import AdaLoraConfig,get_peft_model
        super().load_pretrained()
        targets=self.reference['model_spec']['adapter_targets'];modules=dict(self.model.named_modules())
        if len(targets)!=22 or any(t not in modules or not isinstance(modules[t],torch.nn.Linear) for t in targets):raise ValueError('AdaLoRA target inventory')
        self.model=get_peft_model(self.model,AdaLoraConfig(init_r=ADAPTER['init_r'],target_r=ADAPTER['target_r'],tinit=ADAPTER['tinit'],tfinal=ADAPTER['tfinal'],deltaT=ADAPTER['deltaT'],total_step=ADAPTER['total_step'],lora_alpha=ADAPTER['lora_alpha'],lora_dropout=ADAPTER['lora_dropout'],beta1=ADAPTER['beta1'],beta2=ADAPTER['beta2'],orth_reg_weight=ADAPTER['orth_reg_weight'],target_modules=targets,modules_to_save=['head','classifier'],bias='none',task_type=None),autocast_adapter_dtype=True)
        self.trainable={n:v for n,v in self.model.named_parameters() if v.requires_grad}
        for n,v in self.trainable.items():
            if not (any(x in n for x in ('lora_A.default','lora_B.default','lora_E.default','head.modules_to_save.default.','classifier.modules_to_save.default.'))):raise ValueError('unadmitted AdaLoRA tensor '+n)
            v.data=v.data.float()
        self.initial_frozen_hash=self.frozen_hash();self.initial_trainable_hash=self.trainable_hash();return self
    def batch_loss(self,records,class_weights=None):
        import torch
        loss,tokens=super().batch_loss(records,class_weights)
        penalties=[]
        for n,p in self.trainable.items():
            if 'lora_A.default' in n or 'lora_B.default' in n:
                covariance=p@p.T if 'lora_A.default' in n else p.T@p
                penalties.append(torch.linalg.norm(covariance-torch.eye(covariance.shape[0],device=p.device,dtype=p.dtype),ord='fro'))
        return loss+ADAPTER['orth_reg_weight']*sum(penalties)/len(penalties),tokens
    def allocation(self):
        config=self.model.peft_config['default'];pattern=config.rank_pattern or {}
        active={n:sum(mask) for n,mask in pattern.items()}
        return {'assigned_rank_pattern':pattern,'assigned_rank_per_target':active,'assigned_rank_budget':sum(active.values()) if active else 22*ADAPTER['init_r'],'physical_rank_per_target':ADAPTER['init_r'],'effective_nonzero_singular_values':{n:int((p.detach()!=0).sum()) for n,p in self.trainable.items() if 'lora_E.default' in n}}
    def allocate(self,step):
        self.model.base_model.update_and_allocate(step)
        a=self.allocation()
        if a['assigned_rank_budget']>22*ADAPTER['init_r'] or step>=ADAPTER['total_step']-ADAPTER['tfinal'] and a['assigned_rank_budget']>22*ADAPTER['target_r']:raise ValueError('rank budget exceeded')
        return a
