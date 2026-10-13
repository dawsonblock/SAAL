import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': '3dc04f6f4341c278510997795f0ba72a5a5176baca0d503056996932868cc08d'}, {'name': 'qa-input-part-002.bin', 'size': 24671942, 'sha256': '8a60889da159bf40195831c24ae4453933e05c1ec62566cf2aed78e0531eac63'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='7118a4b6b246b3ac82e88df461579970c40ad0fca019a5ffa8e906f9d7ddce5a':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
