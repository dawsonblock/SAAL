import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/colab-tests-02');session='saal-e91ec137e3f644e39369'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'b72f779d74086061d9e30af045b81813ac797c08ade5a14d97d4af5bbe7c8a8c'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '2e5a3f8b637df792953487fd470f8c45383440c8ecacd6801a20c4acc58cf66e'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '0e408f1d8492bd3a5c17d88c509f158f798fb4ec2d722b170632d5f0453142f4'}, {'name': 'qa-input-part-004.bin', 'size': 8987797, 'sha256': '395c30284fdd87ddfa707d14201371241e84ac90bc49ddcdb5aac4b15799a128'}]:
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
