"""Native model adapters with exact trainable inventories and safe snapshots."""
from abc import ABC,abstractmethod
from pathlib import Path
import hashlib
from .pretrained_contract import PretrainedModelSpec,VERSIONS

class LearningBackend(ABC):
    def __init__(self,spec,snapshot,seed):
        self.spec=PretrainedModelSpec.parse(spec);self.snapshot=Path(snapshot);self.seed=seed;self.model=None;self.tokenizer=None
    def validate_model_spec(self):return self.spec.as_dict()
    @abstractmethod
    def load_pretrained(self):...
    @abstractmethod
    def batch_loss(self,records,class_weights=None):...
    @abstractmethod
    def evaluate(self,records,max_new_tokens=80):...
    def initialize_experiment(self):
        import torch
        from peft import LoraConfig,get_peft_model
        method=self.spec.adapter_type
        for p in self.model.parameters():p.requires_grad_(False)
        if method=='full':
            for p in self.model.parameters():p.requires_grad_(True)
        elif method=='head':
            for n,p in self.model.named_parameters():
                if n.startswith(('head.','classifier.')):p.requires_grad_(True)
        elif method=='lora':
            modules=dict(self.model.named_modules())
            for target in self.spec.adapter_targets:
                if target not in modules or not isinstance(modules[target],torch.nn.Linear):raise ValueError('native adapter target mismatch: '+target)
            config=LoraConfig(r=self.spec.adapter_rank,lora_alpha=2*self.spec.adapter_rank,lora_dropout=.05,target_modules=self.spec.adapter_targets,
                modules_to_save=['head','classifier'] if self.spec.model_kind=='modernbert-seqcls-v1' else None,bias='none',task_type=None)
            self.model=get_peft_model(self.model,config,autocast_adapter_dtype=True)
        self.trainable={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        for n,p in self.trainable.items():
            if 'visual' in n:raise ValueError('vision adaptation prohibited')
            if method=='lora' and not ('lora_A.default.weight' in n or 'lora_B.default.weight' in n or self.spec.model_kind=='modernbert-seqcls-v1' and any(x in n for x in ('head.modules_to_save.default.','classifier.modules_to_save.default.'))):raise ValueError('unadmitted trainable parameter: '+n)
            # Mixed-precision training keeps admitted trainable tensors in FP32.
            p.data=p.data.float()
        if method!='none' and not self.trainable:raise ValueError('empty admitted parameter inventory')
        self.initial_frozen_hash=self.frozen_hash();self.initial_trainable_hash=self.trainable_hash()
        return self.parameter_inventory()
    def tensor_hash(self,items):
        import torch
        h=hashlib.sha256()
        for n,p in sorted(items):
            h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def frozen_hash(self):return self.tensor_hash((n,p) for n,p in self.model.named_parameters() if not p.requires_grad)
    def trainable_hash(self):return self.tensor_hash(self.trainable.items())
    def verify_frozen_weights(self):
        final=self.frozen_hash()
        if final!=self.initial_frozen_hash:raise ValueError('frozen backbone changed')
        return {'initial':self.initial_frozen_hash,'final':final,'unchanged':True,'vision_frozen':self.spec.model_kind=='qwen35-generative-v1','frozen_parameter_count':sum(p.numel() for p in self.model.parameters() if not p.requires_grad),'scope':'Only parameters outside the admitted trainable inventory are claimed unchanged; full tuning changes the backbone.'}
    def parameter_inventory(self):return {n:{'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel()} for n,p in sorted(self.trainable.items())}
    def export_checkpoint(self,path):
        from safetensors.torch import save_file
        import torch
        tensors={n:p.detach().cpu().contiguous() for n,p in self.trainable.items()}
        if not tensors:tensors={'__no_adaptation__':torch.zeros(1)}
        save_file(tensors,str(path));return self.parameter_inventory()
    def restore_checkpoint(self,path):
        from safetensors.torch import load_file
        import torch
        path=Path(path)
        if path.is_symlink() or not path.is_file() or not 1<=path.stat().st_size<=650_000_000:raise ValueError('checkpoint file bound')
        state=load_file(str(path),device='cpu');expected=set(self.trainable) or {'__no_adaptation__'}
        if set(state)!=expected:raise ValueError('checkpoint parameter inventory')
        for n,t in state.items():
            if t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('checkpoint dtype/finite values')
            if n=='__no_adaptation__':
                if t.shape!=(1,) or t.item()!=0:raise ValueError('invalid no-adaptation sentinel')
                continue
            p=self.trainable[n]
            if t.shape!=p.shape:raise ValueError('checkpoint tensor shape')
        # Validate the complete inventory before modifying any live parameter.
        for n,t in state.items():
            if n!='__no_adaptation__':
                with torch.no_grad():self.trainable[n].copy_(t.to(self.trainable[n].device))
    def report_resources(self):
        import torch
        return {'trainable_parameters':sum(p.numel() for p in self.trainable.values()),'total_parameters':sum(p.numel() for p in self.model.parameters()),
          'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
    def train(self,dataset,budget):
        from .peft_training import train_backend
        return train_backend(self,dataset,budget)

def create_backend(spec,snapshot,seed):
    s=PretrainedModelSpec.parse(spec)
    if s.model_kind=='modernbert-seqcls-v1':
        from .modernbert_backend import ModernBERTBackend
        return ModernBERTBackend(spec,snapshot,seed)
    from .qwen35_backend import Qwen35Backend
    return Qwen35Backend(spec,snapshot,seed)

def verify_dependencies():
    import importlib.metadata
    actual={n:importlib.metadata.version(n) for n in VERSIONS}
    if actual!=VERSIONS:raise ValueError('runtime dependency mismatch: '+str(actual))
    return actual
