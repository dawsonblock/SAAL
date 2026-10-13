import sys,time,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.model_monitor import stream_command
from saal_rsi.pretrained_registry import file_hash
workspace=Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/evidence-closure-v5/colab-tests');session='saal-addd0251022d4a9c9f48'
for item in [{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'aebcaebda075c563c87a88efb9f662063c19a8dc56930d133e4e302a299d8eb0'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': 'f21f6db0790e3099d67391eee7e56e6f9fad846e1ad61887c3d78a4aa2627e88'}, {'name': 'qa-input-part-003.bin', 'size': 4169469, 'sha256': 'cdc249623d8fdeb2963eabc9dd7fe7d7561a7424534ca713f392ad09e4708b5a'}]:
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
