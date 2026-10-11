"""Review signatures prove key attribution, not that a human or independent role acted."""
import csv,json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from .policy import canonical,digest,LABELS
from .integrity import validate

def sign_review(corpus,csv_path,reviewer_id,key,role='reviewer'):
 validate(corpus)
 if role not in ('reviewer','adjudicator'):raise ValueError('review role')
 rows=list(csv.DictReader(open(csv_path,newline='')));ids={r['record_id'] for r in corpus['records']};answers={}
 for r in rows:
  if set(r)!={'record_id','prompt','decision','rationale'} or r['record_id'] in answers or r['record_id'] not in ids or r['decision'] not in LABELS+('REJECT',) or not 1<=len(r['rationale'].encode())<=2048:raise ValueError('complete decisions/rationales required')
  original=next(x for x in corpus['records'] if x['record_id']==r['record_id'])
  if r['prompt']!=original['prompt']:raise ValueError('review prompt substitution')
  answers[r['record_id']]={'decision':r['decision'],'rationale':r['rationale']}
 if set(answers)!=ids:raise ValueError('review covers exact full corpus')
 body={'schema':'saal-review-batch-v1','corpus_digest':digest(corpus),'reviewer_id':reviewer_id,'role':role,'decisions':answers}
 return {'payload':body,'signature':key.sign(canonical(body)).hex()}

def verified(envelope,pin):
 try:Ed25519PublicKey.from_public_bytes(bytes.fromhex(pin)).verify(bytes.fromhex(envelope['signature']),canonical(envelope['payload']))
 except Exception as e:raise ValueError('review signature') from e
 return envelope['payload']

def assess(corpus,policy,reviews,adjudication=None,semantic_review=None):
 validate(corpus)
 if policy['schema']!='saal-reviewer-policy-v1' or len(policy['reviewers'])!=2:raise ValueError('two enrolled reviewers required')
 pins=list(policy['reviewers'].values());adj=policy['adjudicator']
 if any(type(p)is not str or len(p)!=64 for p in pins+[adj]) or len(set(pins+[adj]))!=3:raise ValueError('distinct reviewer/adjudicator identities required')
 if len(reviews)!=2:raise ValueError('review quorum missing')
 bodies={}
 for e in reviews:
  identity=e['payload']['reviewer_id']
  if identity in bodies or identity not in policy['reviewers']:raise ValueError('reviewer impersonation/duplication')
  b=verified(e,policy['reviewers'][identity]);_batch(corpus,b,'reviewer');bodies[identity]=b
 if set(bodies)!=set(policy['reviewers']):raise ValueError('review quorum mismatch')
 values=list(bodies.values());adjudicated=None
 if adjudication is not None:
  adjudicated=verified(adjudication,adj);_batch(corpus,adjudicated,'adjudicator')
 decisions={};disagreements=[]
 for r in corpus['records']:
  rid=r['record_id'];a,b=[v['decisions'][rid]['decision'] for v in values]
  if 'REJECT' in (a,b):raise ValueError('reviewer rejected record; repair in a new corpus version')
  if a!=b:
   disagreements.append(rid)
   if adjudicated is None:raise ValueError('unresolved label disagreement')
   label=adjudicated['decisions'][rid]['decision']
  else:label=a
  # Human/oracle disagreements must produce a new version and review cycle,
  # rather than silently relabeling the already-bound candidate.
  if label!=r['proposed_decision']:raise ValueError('human/oracle disagreement requires a new corpus version')
  decisions[rid]=label
 if semantic_review is None:raise ValueError('semantic leakage review missing')
 screen=verified(semantic_review,adj)
 if screen.get('schema')!='saal-semantic-split-review-v1' or screen.get('corpus_digest')!=digest(corpus) or screen.get('approved')is not True or type(screen.get('rationale'))is not str or not screen['rationale'].strip() or screen.get('independence_claim')is not False:raise ValueError('semantic split review invalid or inflated')
 from .leakage import screen as lexical_screen
 lexical=lexical_screen(corpus)
 if screen.get('lexical_screen_digest')!=digest(lexical):raise ValueError('stale/substituted semantic screen review')
 dispositions=screen.get('family_pair_dispositions',{})
 if set(dispositions)!=set(lexical['family_pair_groups']):raise ValueError('unresolved cross-split family review')
 for d in dispositions.values():
  if type(d)is not dict or set(d)!={'decision','rationale'} or d['decision']!='ACCEPT_PUBLIC_SHARED_POLICY' or type(d['rationale'])is not str or not d['rationale'].strip():raise ValueError('semantic rewrite/repartition/review remains required')
 return {'schema':'saal-review-assessment-v1','corpus_digest':digest(corpus),'label_review_status':'SIGNED_QUORUM_COMPLETE','disagreements':disagreements,'semantic_split_review':'SIGNED_CURATOR_REVIEW','review_evidence_digests':[digest(x) for x in reviews]+([digest(adjudication)] if adjudication else [])+[digest(semantic_review)],'labels':decisions,'eligible_for_separate_training_admission':True,'human_identity_and_operational_independence':'EXTERNAL_TRUST_ASSUMPTION_NOT_ESTABLISHED_BY_SIGNATURES','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}

def _batch(corpus,b,role):
 if b.get('schema')!='saal-review-batch-v1' or b.get('role')!=role or b.get('corpus_digest')!=digest(corpus) or set(b.get('decisions',{}))!={r['record_id'] for r in corpus['records']}:raise ValueError('stale/substituted/incomplete review')
 for x in b['decisions'].values():
  if set(x)!={'decision','rationale'} or x['decision'] not in LABELS+('REJECT',) or type(x['rationale'])is not str or not 1<=len(x['rationale'].encode())<=2048:raise ValueError('review decision/rationale')
