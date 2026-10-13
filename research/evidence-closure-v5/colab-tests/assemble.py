import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'aebcaebda075c563c87a88efb9f662063c19a8dc56930d133e4e302a299d8eb0'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': 'f21f6db0790e3099d67391eee7e56e6f9fad846e1ad61887c3d78a4aa2627e88'}, {'name': 'qa-input-part-003.bin', 'size': 4169469, 'sha256': 'cdc249623d8fdeb2963eabc9dd7fe7d7561a7424534ca713f392ad09e4708b5a'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='e43b6f3c5cf56fbaec176438f4b8ae7d170c04e1f97d9e12a837c31b47c0cee3':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
