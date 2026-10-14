"""Versioned provenance and conservative leakage checks; IDs cannot prove independence."""
from dataclasses import dataclass,asdict
from difflib import SequenceMatcher
import re
import unicodedata
from .model_contract import LABELS,CATEGORIES
from .spec import sha

FIELDS={'record_id','task_family_id','cluster_id','template_id','prompt','expected_decision','category','split','source_digest','label_rationale','oracle_version','parent_record_id'}
SPLITS=('train','development','regression','holdout')

@dataclass(frozen=True)
class DatasetRecordV2:
    record_id:str
    task_family_id:str
    cluster_id:str
    template_id:str
    prompt:str
    expected_decision:str
    category:str
    split:str
    source_digest:str
    label_rationale:str
    oracle_version:str
    parent_record_id:str|None=None
    def as_dict(self):return asdict(self)


def identifier(value):return type(value)is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value) is not None

def digest(value):return type(value)is str and re.fullmatch(r'sha256:[a-f0-9]{64}',value) is not None

def normalize(prompt):
    text=unicodedata.normalize('NFKC',prompt).casefold()
    # Quantities may change meaning; retain numbers rather than treating them as duplicates.
    return ' '.join(re.findall(r'\w+',text))


def validate_records(records):
    if type(records)is not list or not 1<=len(records)<=2000:raise ValueError('v2 dataset size')
    ids={};owners={};family_split={};normalized={}
    for r in records:
        if type(r)is not dict or set(r)!=FIELDS:raise ValueError('DatasetRecordV2 schema')
        if any(not identifier(r[k]) for k in ('record_id','task_family_id','cluster_id','template_id','oracle_version')):raise ValueError('v2 record identity')
        if r['record_id'] in ids:raise ValueError('duplicate record identity')
        if type(r['prompt'])is not str or not 1<=len(r['prompt'].encode())<=1024:raise ValueError('v2 prompt bound')
        if type(r['label_rationale'])is not str or not 1<=len(r['label_rationale'].encode())<=2048 or not digest(r['source_digest']):raise ValueError('v2 provenance/rationale')
        if r['expected_decision'] not in LABELS[:4] or r['category'] not in CATEGORIES or r['split'] not in SPLITS:raise ValueError('v2 category/label/split')
        if r['parent_record_id'] is not None and not identifier(r['parent_record_id']):raise ValueError('v2 parent identity')
        family=r['task_family_id'];split=r['split'];ids[r['record_id']]=r
        if family in family_split and family_split[family]!=split:raise ValueError('task-family split leakage')
        family_split[family]=split
        for k in ('cluster_id','template_id','source_digest'):
            key=(k,r[k])
            if key in owners and owners[key]!=family:raise ValueError('generator/provenance family leakage: '+k)
            owners[key]=family
        key=normalize(r['prompt'])
        if key in normalized:raise ValueError('exact/normalized duplicate prompt')
        normalized[key]=r['record_id']
    for r in records:
        seen={r['record_id']};node=r
        while node['parent_record_id'] is not None:
            parent=ids.get(node['parent_record_id'])
            if parent is None:raise ValueError('missing lineage parent')
            if any(parent[k]!=r[k] for k in ('task_family_id','cluster_id','split','source_digest')):raise ValueError('counterfactual/augmentation lineage leakage')
            if parent['record_id'] in seen:raise ValueError('lineage cycle')
            seen.add(parent['record_id']);node=parent
    return records


def manifest(records,*,origin='public-synthetic',sampling_description='Public curated diagnostics; no independent sampling claim'):
    validate_records(records)
    if origin not in ('public-synthetic','operator-supplied','custodian-sequestered'):raise ValueError('dataset origin')
    return {'schema':'saal-dataset-manifest-v2','origin':origin,'records':records,
            'sampling':{'unit':'task_family_id','description':sampling_description,'design_digest':sha({'origin':origin,'description':sampling_description,'family_sources':sorted({(r['task_family_id'],r['source_digest']) for r in records})})}}


def validate_manifest(m):
    if type(m)is not dict or set(m)!={'schema','origin','records','sampling'} or m['schema']!='saal-dataset-manifest-v2':raise ValueError('v2 manifest schema')
    if type(m['sampling'])is not dict or set(m['sampling'])!={'unit','description','design_digest'} or type(m['sampling']['description'])is not str or not 1<=len(m['sampling']['description'])<=4000:raise ValueError('sampling specification')
    if m!=manifest(m['records'],origin=m['origin'],sampling_description=m['sampling']['description']):raise ValueError('dataset manifest identity')
    return m


def leakage_audit(*manifests):
    records=[]
    for m in manifests:validate_manifest(m);records.extend(m['records'])
    validate_records(records) # hard identity, normalization, grouping and lineage constraints
    flags=[]
    for i,left in enumerate(records):
        a=normalize(left['prompt']);ta=set(a.split())
        for right in records[i+1:]:
            if left['task_family_id']==right['task_family_id']:continue
            b=normalize(right['prompt']);tb=set(b.split())
            ratio=SequenceMatcher(None,a,b,autojunk=False).ratio();jaccard=len(ta&tb)/len(ta|tb)
            if ratio>=.88 or jaccard>=.8:
                flags.append({'left':left['record_id'],'right':right['record_id'],'sequence_similarity':round(ratio,6),'token_similarity':round(jaccard,6),
                              'review_reason':'approximate lexical similarity; semantic leakage is undetermined'})
    return {'schema':'saal-dataset-audit-v2','dataset_digests':[sha(m) for m in manifests],'hard_checks_passed':True,'review_flags':flags,
            'independence_established':False,'limits':'Approximate lexical screening is not semantic duplicate detection. A signed independent review is required for protected qualification.'}


def legacy_projection(m,split):
    validate_manifest(m)
    if split not in ('train','development','holdout'):raise ValueError('legacy projection split')
    rows=[r for r in m['records'] if r['split']==split]
    if not rows or any(len(r['prompt'].encode())>128 for r in rows):raise ValueError('historical worker requires nonempty <=128-byte prompts')
    return [{'id':r['record_id'],'cluster':r['cluster_id'],'prompt':r['prompt'],'label':r['expected_decision'],'category':r['category'],'split':split} for r in rows]


def migrate_legacy(records,provenance):
    """Never infer independent families from case numbers or cluster IDs."""
    if type(provenance)is not dict or set(provenance)!={r['id'] for r in records}:raise ValueError('explicit complete migration provenance required')
    result=[]
    for r in records:
        p=provenance[r['id']]
        if type(p)is not dict or set(p)!={'task_family_id','template_id','source_digest','label_rationale','oracle_version','parent_record_id'}:raise ValueError('migration provenance schema')
        result.append(dict(p,record_id=r['id'],cluster_id=r['cluster'],prompt=r['prompt'],expected_decision=r['label'],category=r['category'],split=r['split']))
    return manifest(result,origin='operator-supplied',sampling_description='Migrated legacy records; independence unverified')
