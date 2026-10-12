"""Screening and signed review of provenance; similarity never certifies independence."""
import math
import re
from .spec import sha
from .model_contract import verify


def normalized_hash(prompt):
    return sha(' '.join(prompt.casefold().split()))


def protected_fingerprints(records):
    """Custodian-side index. No answers or raw protected prompts are returned."""
    return {'schema': 'saal-protected-lineage-index-v1',
            'normalized_prompts': sorted({normalized_hash(r['prompt']) for r in records}),
            'record_ids': sorted({r['record_id'] for r in records}),
            'sources': sorted({r['source_digest'] for r in records}),
            'templates': sorted({r['template_id'] for r in records}),
            'mechanisms': sorted({r['generation_mechanism'] for r in records})}


def provenance_screen(records, protected=None):
    if type(records) is not list or not 1 <= len(records) <= 25000:
        raise ValueError('lineage population bound')
    ids = set(); flags = []; denied = []; seen = {}; span_owners = {}
    for r in records:
        required = ('record_id', 'split', 'task_family_id', 'cluster_id', 'source_digest',
                    'template_id', 'generation_mechanism', 'prompt', 'parent_record_id')
        if type(r) is not dict or any(k not in r for k in required) or r['record_id'] in ids:
            raise ValueError('incomplete or duplicate lineage')
        ids.add(r['record_id'])
        if any(type(r[k]) is not str or not r[k] for k in required if k != 'parent_record_id') or len(r['prompt'].encode()) > 4096:
            raise ValueError('lineage field bound')
        for field in ('source_digest', 'template_id', 'generation_mechanism'):
            key = (field, r[field]); other = seen.get(key)
            if other and other['split'] != r['split']:
                flag = {'left': other['record_id'], 'right': r['record_id'], 'reason': field + '_OVERLAP'}
                if len(flags) < 2000: flags.append(flag)
            else: seen[key] = r
        text = normalized_hash(r['prompt'])
        if ('text', text) in seen:
            denied.append({'record': r['record_id'], 'reason': 'NORMALIZED_DUPLICATE'})
        seen[('text', text)] = r
        words = re.findall(r'\w+', r['prompt'].casefold())
        for i in range(len(words) - 4):
            span = tuple(words[i:i + 5]); other = span_owners.get(span)
            if other and other['split'] != r['split'] and len(flags) < 2000:
                flags.append({'left': other['record_id'], 'right': r['record_id'], 'reason': 'LEXICAL_NEAR_DUPLICATE'})
            else: span_owners[span] = r
        if protected is not None:
            if set(protected) != {'schema', 'normalized_prompts', 'record_ids', 'sources', 'templates', 'mechanisms'} or protected['schema'] != 'saal-protected-lineage-index-v1':
                raise ValueError('protected lineage index shape')
            hit = (text in protected['normalized_prompts'] or r['record_id'] in protected['record_ids'] or
                   r['parent_record_id'] in protected['record_ids'] or r['source_digest'] in protected['sources'] or
                   r['template_id'] in protected['templates'] or r['generation_mechanism'] in protected['mechanisms'])
            if hit: denied.append({'record': r['record_id'], 'reason': 'PROTECTED_DERIVATION_OR_MECHANISM'})
    owners = {r['record_id']: r for r in records}
    for r in records:
        if r['parent_record_id'] is not None:
            p = owners.get(r['parent_record_id'])
            if p is None or (p['split'], p['task_family_id'], p['cluster_id']) != (r['split'], r['task_family_id'], r['cluster_id']):
                denied.append({'record': r['record_id'], 'reason': 'PARENT_SPLIT_OR_SOURCE_LEAKAGE'})
    flags = list({sha(flag): flag for flag in flags}.values())
    return {'schema': 'saal-lineage-screen-v1', 'records_digest': sha(records),
            'protected_index_digest': sha(protected) if protected is not None else None,
            'denied': denied, 'review_flags': flags, 'semantic_screening': 'NOT_PERFORMED',
            'independence_established': False, 'qualification_eligible': False,
            'limits': 'Common mechanisms and lexical similarity require review. No embedding score or review signature proves independent sampling.'}


def embedding_screen(records, embeddings, model_identity, *, threshold=.95):
    """Bounded cosine screening over cross-split pairs; no automatic resolutions."""
    from .attempt_authority import digest
    digest(model_identity)
    if type(records) is not list or not 2 <= len(records) <= 2000 or type(embeddings) is not list or len(embeddings) != len(records) or type(threshold) is not float or not .5 <= threshold <= 1:
        raise ValueError('embedding screening bounds')
    width = len(embeddings[0])
    if not 2 <= width <= 2048:
        raise ValueError('embedding dimension')
    vectors = []
    for v in embeddings:
        if type(v) is not list or len(v) != width or any(type(x) not in (int, float) or not math.isfinite(x) for x in v):
            raise ValueError('embedding tensor identity')
        norm = math.sqrt(sum(x * x for x in v))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('embedding norm')
        vectors.append([x / norm for x in v])
    flags = []; total = 0
    for i, a in enumerate(vectors):
        for j in range(i):
            if records[i]['split'] == records[j]['split']: continue
            cosine = max(-1., min(1., sum(x * y for x, y in zip(a, vectors[j]))))
            if cosine >= threshold:
                total += 1
                if len(flags) < 2000:
                    flags.append({'left': records[j]['record_id'], 'right': records[i]['record_id'],
                                  'reason': 'SEMANTIC_SIMILARITY_REQUIRES_REVIEW', 'cosine': cosine})
    return {'schema': 'saal-embedding-screen-v1', 'records_digest': sha(records),
            'embedding_digest': sha(embeddings), 'model_identity': model_identity,
            'threshold': threshold, 'flag_count': total, 'review_flags': flags,
            'truncated': total > len(flags), 'independence_established': False}


def verify_screen_review(envelope, reviewer_pin, screen):
    body = verify(envelope, reviewer_pin)
    if type(body) is not dict or set(body) != {'schema', 'screen_digest', 'resolutions', 'sampling_description', 'scope'} or body['schema'] != 'saal-screen-review-v1' or body['screen_digest'] != sha(screen) or body['scope'] != 'PUBLIC_LINEAGE_REVIEW_ONLY':
        raise ValueError('review identity or scope')
    flags = {sha(f) for f in screen['review_flags']}
    if type(body['resolutions']) is not dict or set(body['resolutions']) != flags or any(type(v) is not str or not 20 <= len(v) <= 2000 for v in body['resolutions'].values()) or screen.get('denied') or screen.get('truncated'):
        raise PermissionError('unresolved, truncated or prohibited leakage')
    if type(body['sampling_description']) is not str or not 40 <= len(body['sampling_description']) <= 8192:
        raise ValueError('sampling description required')
    return {'review_complete': True, 'independence_established': False, 'qualification_eligible': False}
