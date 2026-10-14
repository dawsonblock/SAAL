import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'c6616260f42f6589b3a782d8569682909e910fd6e4054188439735b407c0c2f1'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '43f8f32f9dc09c58996676a9fc2178dbfd933b4a3bdbefadbb2918cf8492504d'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': 'b7b73be48f3c94d6d76082076e13c05a560365427d07ce968ac2c93e1d82fd97'}, {'name': 'qa-input-part-004.bin', 'size': 8987503, 'sha256': 'ebb8fb9e06a1836110a850a2e4dc17b9aa2573dba21c0a903cec7b5c48942537'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='7ee673ff4cdf747dbf94882b04237bc653b85d4dea95fcad0cbdde9e0cb5704c':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
