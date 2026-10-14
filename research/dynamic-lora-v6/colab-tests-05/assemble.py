import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'c6616260f42f6589b3a782d8569682909e910fd6e4054188439735b407c0c2f1'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '27824bd3a6debad82ee33887a90906ee222023df1f180c8965878b907d7498af'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '22077b7726305634653b1c3257c90c66c23053a8818556363ef09fa960b8e37e'}, {'name': 'qa-input-part-004.bin', 'size': 9010446, 'sha256': '58c1a2d677cd385a2e6098d87c2632e7f7117178a8b90d6c165f99ef78c45a46'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='326fd0d864f2f86e70655303d728f53503312467ddb2d676cd29b361dd69ab26':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
