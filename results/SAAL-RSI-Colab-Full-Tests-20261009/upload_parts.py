from pathlib import Path
import zipfile,hashlib,json,subprocess
r=Path(__file__).parent;p=r.parent
with zipfile.ZipFile(p/'SAAL-RSI-v0.9.4.2-Full-Validation-Repairs.zip') as z,zipfile.ZipFile(r/'base-tests.zip','a',zipfile.ZIP_DEFLATED) as out:
 for i in z.infolist():
  if i.filename.startswith('provenance/') and not i.filename.endswith('.zip'):out.writestr(i.filename,z.read(i))
parts=[]
with (r/'base-tests.zip').open('rb') as f:
 n=0
 while data:=f.read(24*1024*1024):
  part=r/f'base.part{n:03d}';part.write_bytes(data);parts.append({'name':part.name,'sha256':hashlib.sha256(data).hexdigest()});n+=1
manifest={'parts':parts,'sha256':hashlib.sha256((r/'base-tests.zip').read_bytes()).hexdigest()};(r/'parts.json').write_text(json.dumps(manifest))
for f in [*(r/x['name'] for x in parts),r/'parts.json']:
 res=subprocess.run(['colab','upload','--session','saal-full-tests-20261009',str(f),'/content/'+f.name],capture_output=True,text=True,timeout=120)
 if res.returncode:raise RuntimeError('Transfer failed for '+f.name+'; raw provider output withheld')
 print('Uploaded '+f.name,flush=True)
