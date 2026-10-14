import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'b72f779d74086061d9e30af045b81813ac797c08ade5a14d97d4af5bbe7c8a8c'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': '2e5a3f8b637df792953487fd470f8c45383440c8ecacd6801a20c4acc58cf66e'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '0e408f1d8492bd3a5c17d88c509f158f798fb4ec2d722b170632d5f0453142f4'}, {'name': 'qa-input-part-004.bin', 'size': 8987797, 'sha256': '395c30284fdd87ddfa707d14201371241e84ac90bc49ddcdb5aac4b15799a128'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='8c94d6a70c4c95e03cb8d2a67bb00eb50569b433cf4d8adc53ca1c5369e44b6c':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
