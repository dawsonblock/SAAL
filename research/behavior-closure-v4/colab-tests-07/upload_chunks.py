import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/behavior-closure-v4/colab-tests-07');session='saal-b9ee4548cdc54fc7a49e'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': '3dc04f6f4341c278510997795f0ba72a5a5176baca0d503056996932868cc08d'}, {'name': 'qa-input-part-002.bin', 'size': 24671942, 'sha256': '8a60889da159bf40195831c24ae4453933e05c1ec62566cf2aed78e0531eac63'}]:
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
