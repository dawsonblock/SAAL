"""Reacquire exact data revisions into a new directory without dataset code execution."""
from pathlib import Path
from urllib.request import Request,urlopen
import json,hashlib,sys
manifest=json.loads((Path(__file__).parent/'ACQUISITION.json').read_text());dest=Path(sys.argv[1]);dest.mkdir(exist_ok=False,parents=True)
for source in manifest['sources']:
 for item in source['files']:
  prefix='sources/'+source['repository'].replace('/','--')+'/'
  assert item['path'].startswith(prefix);name=item['path'][len(prefix):];repo=source['repository'];revision=source['revision']
  url=('https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+name) if source['repository_type']=='dataset' else ('https://raw.githubusercontent.com/'+repo+'/'+revision+'/'+name)
  with urlopen(Request(url,headers={'User-Agent':'SAAL-RSI-public-data-reproduce'}),timeout=120) as f:data=f.read()
  assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'source substitution'
  path=dest/item['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(dest/'ACQUISITION.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Pinned data reconstructed; no SAAL training or qualification granted')
