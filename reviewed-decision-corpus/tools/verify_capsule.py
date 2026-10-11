"""Verify immutable projections and source bindings; no review/training approval."""
import csv, hashlib, json, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from saal_corpus.build import build
from saal_corpus.integrity import validate
from saal_corpus.policy import canonical,digest

def verify(root=ROOT):
 c=json.loads((root/'corpus/CORPUS.json').read_text());result=validate(c)
 assert (root/'corpus/CORPUS_DIGEST.txt').read_text().strip()==digest(c)
 for split in ['train','validation','test']:
  rows=[json.loads(line) for line in (root/f'corpus/{split}.jsonl').read_text().splitlines()]
  assert rows==[r for r in c['records'] if r['split']==split],f'{split} projection changed'
 for reviewer in ['reviewer_1','reviewer_2']:
  with (root/f'corpus/{reviewer}-blind.csv').open(newline='') as f:rows=list(csv.DictReader(f))
  assert rows==[{'record_id':r['record_id'],'prompt':r['prompt'],'decision':'','rationale':''} for r in c['records']]
 b=json.loads((root/'SOURCE_BINDINGS.json').read_text());assert b['corpus_digest']==digest(c)
 for path,sha in b['source_files'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha,path
 with tempfile.TemporaryDirectory() as t:assert canonical(build(t))==canonical(c),'generation drift'
 result.update({'projection_checks':'PASS','blank_review_forms':'PASS','source_bindings':'PASS','deterministic_rebuild':'PASS','human_reviews_completed':0})
 return result
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True,indent=2))
