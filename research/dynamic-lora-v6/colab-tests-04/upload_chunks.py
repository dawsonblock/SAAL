import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/colab-tests-04');session='saal-51395207ee104375a0d2'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'c6616260f42f6589b3a782d8569682909e910fd6e4054188439735b407c0c2f1'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '43f8f32f9dc09c58996676a9fc2178dbfd933b4a3bdbefadbb2918cf8492504d'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': 'b7b73be48f3c94d6d76082076e13c05a560365427d07ce968ac2c93e1d82fd97'}, {'name': 'qa-input-part-004.bin', 'size': 8987503, 'sha256': 'ebb8fb9e06a1836110a850a2e4dc17b9aa2573dba21c0a903cec7b5c48942537'}]:
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
