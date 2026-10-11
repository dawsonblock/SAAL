import sys,json,platform,subprocess,importlib.metadata
from pathlib import Path
import torch

def test_cuda_adalora_update_allocation_and_safe_reload(tmp_path):
 from torch import nn
 from peft import AdaLoraConfig,get_peft_model
 from safetensors.torch import save_file,load_file
 assert torch.cuda.is_available();assert 'T4' in torch.cuda.get_device_name(0)
 torch.manual_seed(47)
 class Net(nn.Module):
  def __init__(self):super().__init__();self.query=nn.Linear(8,8);self.value=nn.Linear(8,4)
  def forward(self,x):return self.value(torch.tanh(self.query(x)))
 m=get_peft_model(Net(),AdaLoraConfig(init_r=4,target_r=1,total_step=12,tinit=2,tfinal=2,deltaT=1,target_modules=['query','value'],lora_alpha=8,lora_dropout=0)).cuda()
 frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
 for step in range(1,13):
  opt.zero_grad();loss=m(torch.randn(8,8,device='cuda')).square().mean();assert torch.isfinite(loss);loss.backward();opt.step();m.base_model.update_and_allocate(step)
 assert sum(sum(v) for v in m.peft_config['default'].rank_pattern.values())<=2
 assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if n in frozen)
 assert any(not torch.equal(p.detach().cpu(),initial[n]) for n,p in m.named_parameters() if n in initial)
 m.eval();x=torch.randn(2,8,device='cuda');expected=m(x).detach();weights={n:p.detach().cpu().contiguous() for n,p in m.named_parameters() if p.requires_grad};save_file(weights,tmp_path/'adapter.safetensors')
 with torch.no_grad():
  for n,p in m.named_parameters():
   if p.requires_grad:p.add_(.1)
  for n,t in load_file(tmp_path/'adapter.safetensors').items():dict(m.named_parameters())[n].copy_(t.to('cuda'))
 assert torch.equal(m(x),expected)

def test_cuda_modernbert_forward_backward():
 from transformers import ModernBertConfig,ModernBertForSequenceClassification
 assert torch.cuda.is_available()
 c=ModernBertConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,max_position_embeddings=64,num_labels=4,local_attention=32,global_attn_every_n_layers=2,_attn_implementation='eager')
 m=ModernBertForSequenceClassification(c).cuda();x=torch.randint(1,64,(2,16),device='cuda');out=m(input_ids=x,attention_mask=torch.ones_like(x),labels=torch.tensor([0,1],device='cuda'));assert torch.isfinite(out.loss);out.loss.backward();assert any(p.grad is not None for p in m.parameters());torch.cuda.synchronize()

if __name__=='__main__':
 assert sys.argv[1]=='--preflight';assert torch.cuda.is_available();data={'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda,'packages':{p:importlib.metadata.version(p) for p in ['torch','peft','transformers','safetensors','numpy','pytest']},'scope':'Full inherited suite plus bounded synthetic CUDA probes; no pretrained campaign or independent qualification'};Path(sys.argv[2]).write_text(json.dumps(data,indent=2));print(json.dumps(data))
