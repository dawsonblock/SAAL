"""Acquire pinned public data only. Never execute downloaded dataset code."""
from pathlib import Path
from urllib.request import Request,urlopen
import json,hashlib,time
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(__file__).parent
if (ROOT/'ACQUISITION.json').exists():raise RuntimeError('Acquisition already frozen; use reproduce.py with a fresh destination')
api=HfApi();sources=[]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def get_json(url):
 with urlopen(Request(url,headers={'User-Agent':'SAAL-RSI-public-data-research'}),timeout=60) as f:return json.load(f)
def acquire_dataset(repo,selector):
 info=api.dataset_info(repo);revision=info.sha;dest=ROOT/'sources'/repo.replace('/','--');dest.mkdir(parents=True,exist_ok=True);files=[]
 for item in info.siblings:
  if not selector(item.rfilename):continue
  path=Path(hf_hub_download(repo,item.rfilename,repo_type='dataset',revision=revision,local_dir=dest));files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'bytes':path.stat().st_size})
 card=info.card_data.to_dict() if info.card_data else {}
 record={'repository':repo,'revision':revision,'repository_type':'dataset','license_metadata':card.get('license'),'files':files,'exposure':'PUBLIC','saal_labels_present':False,'qualification_allowed':False};sources.append(record);print(repo,revision,len(files),'files',flush=True)
acquire_dataset('google/IFEval',lambda p:p in ['README.md','ifeval_input_data.jsonl'])
acquire_dataset('facebook/anli',lambda p:p=='README.md' or p.endswith('.parquet'))
repo='allenai/natural-instructions';info=get_json('https://api.github.com/repos/'+repo+'/commits/HEAD');revision=info['sha'];tree=get_json('https://api.github.com/repos/'+repo+'/git/trees/'+revision+'?recursive=1');assert not tree.get('truncated');paths=[x['path'] for x in tree['tree'] if x['type']=='blob'];tasks=[p for p in paths if p.startswith('tasks/') and p.endswith('.json')];dest=ROOT/'sources/allenai--natural-instructions';dest.mkdir(parents=True,exist_ok=True)
(ROOT/'NI_TASK_INDEX.json').write_text(json.dumps({'revision':revision,'paths':tasks,'selection':'First lexicographically matching path per declared keyword, without task-performance selection'},indent=2)+'\n')
keywords=['entailment','question','summarization','classification','extraction','translation','paraphras','sentiment','coreference','commonsense','rewrite','reasoning'];selected=[]
for word in keywords:
 candidates=sorted(p for p in tasks if word in p.lower() and p not in selected)
 if candidates:selected.append(candidates[0])
files=[]
for name in [p for p in paths if p in ['README.md','LICENSE','LICENSE.md']]+selected:
 url='https://raw.githubusercontent.com/'+repo+'/'+revision+'/'+name
 with urlopen(Request(url,headers={'User-Agent':'SAAL-RSI-public-data-research'}),timeout=60) as f:data=f.read()
 path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'bytes':len(data)})
sources.append({'repository':repo,'revision':revision,'repository_type':'github','task_index_count':len(tasks),'downloaded_tasks':selected,'files':files,'license_metadata':'See upstream LICENSE and per-task source/license metadata','exposure':'PUBLIC','saal_labels_present':False,'qualification_allowed':False});print(repo,revision,len(selected),'selected tasks',flush=True)
models=[]
for repo in ['answerdotai/ModernBERT-base','Qwen/Qwen3.5-0.8B-Base','Qwen/Qwen3.5-0.8B']:
 info=api.model_info(repo,revision='11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88' if 'ModernBERT' in repo else None);card=info.card_data.to_dict() if info.card_data else {};models.append({'repository':repo,'revision':info.sha,'license_metadata':card.get('license'),'weights_acquired_by_this_capsule':False,'training_admitted':False,'reason':'Repository pin alone is insufficient; actual files, tokenizer, worker, dependencies, adapters and budget must be authenticated separately'})
(ROOT/'ACQUISITION.json').write_text(json.dumps({'schema':'saal-public-data-acquisition-v1','sources':sources,'model_references':models,'qualification_status':'NOT_EVALUATED','saal_training_admitted':False},indent=2)+'\n')
