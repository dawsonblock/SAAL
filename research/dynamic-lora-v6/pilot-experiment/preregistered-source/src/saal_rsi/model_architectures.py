"""Small supervised byte Transformer. This is not a pretrained language model."""
import torch
from torch import nn
from torch.nn import functional as F
from .neural_contract import validate_spec

PAD=256
CLS=257

class LoRALinear(nn.Module):
    def __init__(self,dim,rank,scale):
        super().__init__();self.base=nn.Linear(dim,dim)
        self.A=nn.Parameter(torch.empty(rank,dim));self.B=nn.Parameter(torch.zeros(dim,rank))
        nn.init.kaiming_uniform_(self.A,a=5**.5);self.scale=scale
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale

class Encoder(nn.Module):
    def __init__(self,s):
        super().__init__();d=s['hidden_dimension'];a=s['adapter_configuration'];self.heads=s['attention_heads'];self.p=s['dropout']
        self.query=LoRALinear(d,a['rank'],a['scale']);self.key=nn.Linear(d,d)
        self.value=LoRALinear(d,a['rank'],a['scale']);self.output=nn.Linear(d,d)
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff1=nn.Linear(d,s['feedforward_dimension']);self.ff2=nn.Linear(s['feedforward_dimension'],d)
        self.drop=nn.Dropout(self.p)
    def forward(self,x,valid):
        b,n,d=x.shape
        def heads(v):return v.reshape(b,n,self.heads,d//self.heads).transpose(1,2)
        q,k,v=(heads(layer(x)) for layer in (self.query,self.key,self.value))
        # True means participate in SDPA; padding keys are excluded. Evaluation dropout must be explicit zero.
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=self.p if self.training else 0.0)
        x=self.norm1(x+self.drop(self.output(attention.transpose(1,2).reshape(b,n,d))))
        return self.norm2(x+self.drop(self.ff2(self.drop(F.gelu(self.ff1(x))))))

class TinyTransformer(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=validate_spec(spec)
        self.embedding=nn.Embedding(258,spec['hidden_dimension'],padding_idx=PAD)
        self.position=nn.Embedding(spec['max_sequence_bytes']+1,spec['hidden_dimension'])
        self.encoder=nn.ModuleList(Encoder(spec) for _ in range(spec['encoder_layers']))
        self.head=nn.Linear(spec['hidden_dimension'],4)
    def forward(self,tokens,mask):
        if tokens.ndim!=2 or tokens.shape!=mask.shape or mask.dtype!=torch.bool or tokens.shape[1]>self.spec['max_sequence_bytes']+1:raise ValueError('token/mask inventory')
        if not torch.all(tokens[:,0]==CLS) or not torch.all(mask[:,0]):raise ValueError('classification token')
        x=self.embedding(tokens)+self.position(torch.arange(tokens.shape[1],device=tokens.device))[None,:,:]
        for layer in self.encoder:x=layer(x,mask)
        return self.head(x[:,0,:])

def tokenize(prompts,spec,*,pad_to=None,device='cpu'):
    validate_spec(spec)
    if type(prompts)is not list or not prompts:raise ValueError('nonempty prompt batch required')
    encoded=[]
    for prompt in prompts:
        if type(prompt)is not str or not 1<=len(prompt.encode('utf-8'))<=spec['max_sequence_bytes']:raise ValueError('UTF-8 input bound; truncation prohibited')
        encoded.append([CLS,*prompt.encode('utf-8')])
    length=max(map(len,encoded)) if pad_to is None else pad_to
    if type(length)is not int or not max(map(len,encoded))<=length<=spec['max_sequence_bytes']+1:raise ValueError('padding bound')
    tokens=torch.full((len(encoded),length),PAD,dtype=torch.long,device=device)
    mask=torch.zeros_like(tokens,dtype=torch.bool)
    for i,values in enumerate(encoded):tokens[i,:len(values)]=torch.tensor(values,device=device);mask[i,:len(values)]=True
    return tokens,mask

def adapter_name(name):return name.endswith(('.query.A','.query.B','.value.A','.value.B'))

def configure_training(model,mode):
    if mode not in ('frozen','full','lora'):raise ValueError('training mode')
    for name,p in model.named_parameters():p.requires_grad_(mode=='full' and not adapter_name(name) or mode=='lora' and adapter_name(name))
    return [p for p in model.parameters() if p.requires_grad]

def initialized(spec):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec['initialization_seed']);return TinyTransformer(spec)

def state_inventory(spec):
    return {name:(tuple(t.shape),t.dtype) for name,t in initialized(spec).state_dict().items()}

def validate_tensors(state,spec,kind='full'):
    inventory=state_inventory(spec)
    expected={k:v for k,v in inventory.items() if kind=='full' or (kind=='adapter')==adapter_name(k)}
    if set(state)!=set(expected):raise ValueError('tensor names')
    for name,t in state.items():
        shape,dtype=expected[name]
        if tuple(t.shape)!=shape or t.dtype!=dtype or not torch.isfinite(t).all():raise ValueError('tensor shape/dtype/finiteness: '+name)
    return state
