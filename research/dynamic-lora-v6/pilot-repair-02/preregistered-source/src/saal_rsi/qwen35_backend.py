"""Official multimodal checkpoint, text-only input. Vision never receives updates."""
from .learning_backend import LearningBackend
from .spec import canonical

SYSTEM='Solve the synthetic task. Return only the requested JSON object, without commentary. Text output cannot authorize real actions.'
def completion_tokens(tokenizer,prompt,output,max_tokens):
    # Apply one native chat template, tokenize the complete sequence once, and
    # require exact prompt-prefix parity before masking. Avoid BPE-boundary leaks.
    prefix=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    text=prefix+canonical(output).decode()+tokenizer.eos_token
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    if ids[:len(prompt_ids)]!=prompt_ids:raise ValueError('completion token boundary mismatch')
    if not prompt_ids or len(ids)>max_tokens or len(ids)==len(prompt_ids):raise ValueError('completion input/token budget')
    return ids,[-100]*len(prompt_ids)+ids[len(prompt_ids):]

class Qwen35Backend(LearningBackend):
    def load_pretrained(self):
        import torch
        from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
        torch.manual_seed(self.seed)
        self.tokenizer=AutoTokenizer.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=False)
        if not self.tokenizer.chat_template:raise ValueError('native chat template absent')
        self.model=Qwen3_5ForConditionalGeneration.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True,
            dtype=torch.float16 if self.spec.precision=='float16' else torch.float32,attn_implementation=self.spec.attention_implementation)
        self.model.to('cuda' if torch.cuda.is_available() and self.spec.compute_backend=='colab' else 'cpu');self.model.config.use_cache=False
        self.initialize_experiment();return self
    def batch_loss(self,records,class_weights=None):
        import torch
        if len(records)!=1:raise ValueError('Qwen microbatch must be one')
        r=records[0];ids,labels=completion_tokens(self.tokenizer,r['prompt'],r['expected_output'],self.spec.max_input_tokens)
        device=next(self.model.parameters()).device
        x=torch.tensor([ids],device=device);y=torch.tensor([labels],device=device)
        out=self.model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False)
        return out.loss.float(),len(ids)
    def evaluate(self,records,max_new_tokens=80):
        import torch
        self.model.eval();result=[]
        with torch.inference_mode():
            for r in records:
                ids=self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
                if not isinstance(ids,torch.Tensor):ids=ids['input_ids']
                if ids.shape[-1]>self.spec.max_input_tokens:raise ValueError('generation prompt limit; truncation forbidden')
                ids=ids.to(next(self.model.parameters()).device)
                y=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
                result.append(self.tokenizer.decode(y[0,ids.shape[-1]:],skip_special_tokens=True))
        return result
