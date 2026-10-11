from pathlib import Path
import argparse,json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .policy import canonical,digest
from .integrity import validate
from .review import assess,sign_review
from .metrics import metrics
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
s=sub.add_parser('check');s.add_argument('corpus')
s=sub.add_parser('sign-review');s.add_argument('corpus');s.add_argument('csv');s.add_argument('--reviewer-id',required=True);s.add_argument('--private-key',required=True);s.add_argument('--role',choices=['reviewer','adjudicator'],default='reviewer');s.add_argument('--output',required=True)
for name in ['assess','prepare-data']:
 s=sub.add_parser(name);s.add_argument('corpus');s.add_argument('--policy',required=True);s.add_argument('--review',action='append',required=True);s.add_argument('--adjudication');s.add_argument('--semantic-review',required=True);s.add_argument('--output',required=True)
s=sub.add_parser('sign-semantic-review');s.add_argument('corpus');s.add_argument('body');s.add_argument('--private-key',required=True);s.add_argument('--output',required=True)
s=sub.add_parser('metrics');s.add_argument('corpus');s.add_argument('predictions');s.add_argument('--split',choices=['train','validation','test'],required=True)
a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());c=read(a.corpus)
if a.command=='check':result=validate(c)
elif a.command=='sign-review':result=sign_review(c,a.csv,a.reviewer_id,Ed25519PrivateKey.from_private_bytes(Path(a.private_key).read_bytes()),a.role)
elif a.command in ('assess','prepare-data'):
 if a.command=='prepare-data':
  from .export import prepare as action
 else:action=assess
 result=action(c,read(a.policy),[read(f) for f in a.review],read(a.adjudication) if a.adjudication else None,read(a.semantic_review))
elif a.command=='sign-semantic-review':
 validate(c);body=read(a.body)
 if body.get('schema')!='saal-semantic-split-review-v1' or body.get('corpus_digest')!=digest(c):raise ValueError('semantic review corpus binding')
 key=Ed25519PrivateKey.from_private_bytes(Path(a.private_key).read_bytes());result={'payload':body,'signature':key.sign(canonical(body)).hex()}
else:
 validate(c);result=metrics([r for r in c['records'] if r['split']==a.split],c['pairs'],read(a.predictions))
if getattr(a,'output',None):
 with Path(a.output).open('xb') as f:f.write(canonical(result))
else:print(json.dumps(result,indent=2))
