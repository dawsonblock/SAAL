"""Official native sequence classification; no token-type IDs or truncation."""
from .learning_backend import LearningBackend
from .pretrained_contract import DECISIONS

class ModernBERTBackend(LearningBackend):
    def load_pretrained(self):
        import torch
        from transformers import AutoTokenizer,ModernBertForSequenceClassification
        torch.manual_seed(self.seed)
        self.tokenizer=AutoTokenizer.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=False)
        self.model=ModernBertForSequenceClassification.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True,num_labels=4,
            id2label=dict(enumerate(DECISIONS)),label2id={d:i for i,d in enumerate(DECISIONS)},dtype=torch.float32,attn_implementation=self.spec.attention_implementation)
        self.model.to('cuda' if torch.cuda.is_available() and self.spec.compute_backend=='colab' else 'cpu')
        self.initialize_experiment();return self
    def tokenize(self,records):
        import torch
        prompts=[r['prompt'] for r in records]
        tokens=self.tokenizer(prompts,padding=True,truncation=False,return_tensors='pt',return_token_type_ids=False)
        if tokens['input_ids'].shape[1]>self.spec.max_input_tokens:raise ValueError('native tokenizer input exceeds admitted limit; truncation forbidden')
        return {k:v.to(next(self.model.parameters()).device) for k,v in tokens.items() if k in ('input_ids','attention_mask')}
    def batch_loss(self,records,class_weights=None):
        import torch
        logits=self.model(**self.tokenize(records)).logits
        labels=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records],device=logits.device)
        return torch.nn.functional.cross_entropy(logits.float(),labels,weight=class_weights),int(self.tokenize(records)['attention_mask'].sum())
    def evaluate(self,records,max_new_tokens=80):
        import torch
        self.model.eval();result=[]
        with torch.inference_mode():
            for i in range(0,len(records),16):
                batch=self.tokenize(records[i:i+16]);logits=self.model(**batch).logits.float()
                if not torch.isfinite(logits).all():raise ValueError('nonfinite classification logits')
                for p in logits.softmax(-1).cpu().tolist():result.append({'decision':DECISIONS[max(range(4),key=lambda j:p[j])],'probabilities':p})
        return result
