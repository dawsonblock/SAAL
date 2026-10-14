import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/colab-tests-05');session='saal-80501b2e08954a61b11f'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'c6616260f42f6589b3a782d8569682909e910fd6e4054188439735b407c0c2f1'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '27824bd3a6debad82ee33887a90906ee222023df1f180c8965878b907d7498af'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '22077b7726305634653b1c3257c90c66c23053a8818556363ef09fa960b8e37e'}, {'name': 'qa-input-part-004.bin', 'size': 9010446, 'sha256': '58c1a2d677cd385a2e6098d87c2632e7f7117178a8b90d6c165f99ef78c45a46'}]:
 path=workspace/item['name']
 if path.stat().st_size!=item['size'] or file_hash(path)!=item['sha256']:raise ValueError('local QA chunk changed')
 for attempt in range(1,6):
  try:
   stream_command(['colab','upload','-s',session,str(path),'/content/'+item['name']],workspace/(item['name']+f'-upload-{attempt}.log'),90)
   break
  except (RuntimeError,TimeoutError):
   if attempt==5:raise
   time.sleep(min(10,2**attempt))
 print(json.dumps({'monitor':'qa_chunk_uploaded','name':item['name'],'bytes':item['size']}),flush=True)
