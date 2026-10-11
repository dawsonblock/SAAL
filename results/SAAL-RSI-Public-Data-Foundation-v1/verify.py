"""Fail-closed public-source integrity checks; never grants training authority."""
from pathlib import Path,PurePosixPath
import hashlib,json,re

def verify(root):
 root=Path(root).resolve();manifest=json.loads((root/'ACQUISITION.json').read_text())
 if manifest['schema']!='saal-public-data-acquisition-v1' or manifest['qualification_status']!='NOT_EVALUATED' or manifest['saal_training_admitted']is not False:raise ValueError('acquisition cannot grant qualification or training')
 seen=set();count=0
 for source in manifest['sources']:
  if not re.fullmatch('[a-f0-9]{40}',source['revision']):raise ValueError('immutable revision required')
  if source['exposure']!='PUBLIC' or source['qualification_allowed']is not False or source['saal_labels_present']is not False:raise ValueError('public supplementary tasks cannot grant SAAL qualification')
  for item in source['files']:
   name=item['path'];relative=PurePosixPath(name);path=root/name
   if name in seen or relative.is_absolute() or '..' in relative.parts or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('invalid file inventory')
   seen.add(name)
   if path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('source substitution')
   count+=1
 for model in manifest['model_references']:
  if not re.fullmatch('[a-f0-9]{40}',model['revision']) or model['training_admitted']is not False or model['weights_acquired_by_this_capsule']is not False:raise ValueError('reference is not admitted model identity')
 return {'source_files_verified':count,'qualification_status':'NOT_EVALUATED','saal_training_admitted':False}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).parent)))
