"""Public train/development projection. Regression bytes never enter a job."""
import re
from collections import Counter

from .learning_dataset import FIELDS, validate_dataset
from .pretrained_contract import DECISIONS
from .spec import canonical, sha
from .task_oracles import oracle


def training_projection(population):
    validate_dataset(population, kind='modernbert-seqcls-v1')
    data = {k: v for k, v in population.items() if k not in ('schema', 'records')}
    data.update(schema='saal-adaptive-public-data-v1',
                records=[r for r in population['records'] if r['split'] in ('train', 'development')],
                parent_population_digest=sha(population))
    return validate_training_data(data)


def validate_training_data(data):
    expected = {'schema', 'origin', 'sampling_description', 'records',
                'independent_population', 'review_status', 'parent_population_digest'}
    if type(data) is not dict or set(data) != expected or data['schema'] != 'saal-adaptive-public-data-v1':
        raise ValueError('train/development-only dataset schema')
    if (data['origin'] != 'public-synthetic' or data['independent_population'] is not False
            or data['review_status'] != 'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED'):
        raise ValueError('public diagnostics cannot claim independent review')
    if type(data['sampling_description']) is not str or not 1 <= len(data['sampling_description'].encode()) <= 8192:
        raise ValueError('bounded sampling description')
    if type(data['parent_population_digest']) is not str or not re.fullmatch('sha256:[0-9a-f]{64}', data['parent_population_digest']):
        raise ValueError('parent population identity')
    records = data['records']
    if type(records) is not list or not 8 <= len(records) <= 25000 or len(canonical(data)) > 30_000_000:
        raise ValueError('dataset bound')
    ids, texts, owners = {}, set(), {}
    for r in records:
        if type(r) is not dict or set(r) != FIELDS or r['split'] not in ('train', 'development') or r['task_type'] != 'decision':
            raise ValueError('protected or regression data / non-classifier task forbidden')
        for k in ('record_id', 'task_family_id', 'cluster_id', 'template_id', 'oracle_version', 'generation_mechanism', 'category'):
            if type(r[k]) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,100}', r[k]):
                raise ValueError('record identity')
        if r['record_id'] in ids or type(r['prompt']) is not str or not 1 <= len(r['prompt'].encode()) <= 4096:
            raise ValueError('unique bounded record')
        ids[r['record_id']] = r
        text = ' '.join(r['prompt'].casefold().split())
        if text in texts:
            raise ValueError('normalized duplicate')
        texts.add(text)
        if (type(r['source_digest']) is not str or not re.fullmatch('sha256:[0-9a-f]{64}', r['source_digest'])
                or r['review_status'] != data['review_status'] or r['difficulty'] not in ('basic', 'counterfactual')
                or type(r['label_rationale']) is not str or not 1 <= len(r['label_rationale'].encode()) <= 4096):
            raise ValueError('provenance / review')
        for key in ('task_family_id', 'cluster_id', 'template_id', 'source_digest'):
            identity = key, r[key]
            owner = r['split'], r['task_family_id']
            if identity in owners and owners[identity] != owner:
                raise ValueError('family, recipe or source leakage')
            owners[identity] = owner
        expected_output = oracle('decision', r['oracle_payload'])
        if r['expected_output'] != expected_output or r['expected_decision'] != expected_output['decision']:
            raise ValueError('label / oracle substitution')
    for r in records:
        parent = r['parent_record_id']
        seen = {r['record_id']}
        while parent is not None:
            if type(parent) is not str or parent not in ids or parent in seen:
                raise ValueError('missing or cyclic lineage')
            seen.add(parent)
            p = ids[parent]
            if any(p[k] != r[k] for k in ('split', 'task_family_id', 'cluster_id')):
                raise ValueError('parent leakage')
            parent = p['parent_record_id']
    if {r['split'] for r in records} != {'train', 'development'}:
        raise ValueError('both public training partitions required')
    for split in ('train', 'development'):
        counts = Counter(r['expected_decision'] for r in records if r['split'] == split)
        if set(counts) != set(DECISIONS) or split == 'train' and len(set(counts.values())) != 1:
            raise ValueError('balanced training / four-class coverage required')
    return data
