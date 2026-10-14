import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/colab-tests-03');session='saal-172bd774e7e64cb1b011'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'b72f779d74086061d9e30af045b81813ac797c08ade5a14d97d4af5bbe7c8a8c'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': 'bc7b439a0ec34407d42b18afa129b4abca2e217c32d9bfdc49e09893f1c01fab'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '2f7205aef50e3469cabbd6127ebef9ed132de4126b2c3094b36e6c1e1d929448'}, {'name': 'qa-input-part-004.bin', 'size': 8990832, 'sha256': 'f75ef843d7d50aeb0fe2abd7c4f19ab17ad14bc7d49f9e0c3fb9cd693c4f214d'}]:
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
