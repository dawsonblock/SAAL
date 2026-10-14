import hashlib,json
from pathlib import Path
parts=[{'name': 'qa-input-part-000.bin', 'size': 32000000, 'sha256': '13b95adee71aeb334ef0d7bac50d70ee845e4cebb02ebf4e7a7231dfbbe7506d'}, {'name': 'qa-input-part-001.bin', 'size': 32000000, 'sha256': 'b72f779d74086061d9e30af045b81813ac797c08ade5a14d97d4af5bbe7c8a8c'}, {'name': 'qa-input-part-002.bin', 'size': 32000000, 'sha256': 'bc7b439a0ec34407d42b18afa129b4abca2e217c32d9bfdc49e09893f1c01fab'}, {'name': 'qa-input-part-003.bin', 'size': 32000000, 'sha256': '2f7205aef50e3469cabbd6127ebef9ed132de4126b2c3094b36e6c1e1d929448'}, {'name': 'qa-input-part-004.bin', 'size': 8990832, 'sha256': 'f75ef843d7d50aeb0fe2abd7c4f19ab17ad14bc7d49f9e0c3fb9cd693c4f214d'}]
whole=hashlib.sha256()
with Path('/content/saal-qa-inputs.zip').open('xb') as target:
 for item in parts:
  p=Path('/content')/item['name']
  with p.open('rb') as source:
   h=hashlib.sha256();size=0
   for data in iter(lambda:source.read(1024*1024),b''):
    target.write(data);whole.update(data);h.update(data);size+=len(data)
  if size!=item['size'] or h.hexdigest()!=item['sha256']:raise ValueError('QA upload chunk substitution')
if whole.hexdigest()!='17a47947530796806ca531343baa9088682f794c84baba0a6edfff6e3d028dfc':raise ValueError('QA whole upload substitution')
print(json.dumps({'monitor':'qa_input_reconstruction','whole_sha256':whole.hexdigest(),'chunks':len(parts)}),flush=True)
