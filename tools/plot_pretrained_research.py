"""Standalone research figures from the published factual JSON summaries."""
from pathlib import Path
import argparse,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary');p.add_argument('--sequences',nargs=2);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
if a.sequences:
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for ax,path,title in zip(axes,a.sequences,['Shared LoRA: no replay','Shared LoRA: 25% replay']):
  r=json.loads(Path(path).read_text());m=r['accuracy_matrix'];im=ax.imshow(m,vmin=0,vmax=1,cmap='viridis')
  ax.set_xticks(range(3),['Preferences /\ncorrections','Refusal /\nclarification','Injection /\nproxy conflicts']);ax.set_yticks(range(3),['After task 1','After task 2','After task 3']);ax.set_title(title)
  for i,row in enumerate(m):
   for j,v in enumerate(row):ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v<.65 else 'black')
 fig.colorbar(im,ax=axes,fraction=.025,label='Public task accuracy');fig.suptitle('Public retention pilot: one seed, correlated recipes')
 for suffix in ('png','pdf'):fig.savefig(out/f'public-retention.{suffix}',bbox_inches='tight')
 plt.close(fig)
if a.summary:
 data=json.loads(Path(a.summary).read_text());groups=[(k,g) for k,g in data['distributions'].items() if g['n']==5]
 fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
 labels=[]
 for i,(k,g) in enumerate(groups):
  name=g['repository'].split('/')[-1];method=g['method'];labels.append(name+'\n'+method)
  for d,field,color,label in [(-.22,'initial_accuracy','#64748b','Initial'),(0,'selected_accuracy','#0284c7','Selected'),(.22,'terminal_accuracy','#059669','Terminal')]:
   v=g[field];ax.bar(i+d,100*v['mean'],width=.21,color=color,label=label if i==0 else None,yerr=100*(v['sample_sd'] or 0),capsize=3)
 ax.set_xticks(range(len(labels)),labels);ax.set_ylim(0,100);ax.set_ylabel('Public development accuracy (%)');ax.set_title('Five-seed descriptive results: mean and sample standard deviation');ax.legend(ncol=3)
 fig.text(.02,-.02,'Different task populations across model kinds. Public correlated pilots; no independent qualification.',fontsize=9)
 for suffix in ('png','pdf'):fig.savefig(out/f'public-learning.{suffix}',bbox_inches='tight')
 plt.close(fig)
