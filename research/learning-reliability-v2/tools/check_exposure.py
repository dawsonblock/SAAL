"""Reconstruct bounded sample/token exposure; consistency is not attestation."""
from pathlib import Path
import collections,json,sys
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.spec import sha

ROOT=Path(__file__).resolve().parents[1]

def reconstruct(cache_root):
    from transformers import AutoTokenizer
    root=ROOT/'experiment';pin=(root/'controller.pub').read_text().strip()
    protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin)
    data=json.loads((ROOT/'corpus/dataset.json').read_text())
    if sha(data)!=protocol['dataset_digest']:raise ValueError('dataset substitution')
    tokenizer_path=Path(cache_root)/'ModernBERT-base/11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88'
    validate_snapshot(tokenizer_path,'answerdotai/ModernBERT-base','11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    train=[r for r in data['records'] if r['split']=='train'];by_id={r['record_id']:r for r in train};lengths={}
    for i in range(0,len(train),64):
        batch=train[i:i+64];tokens=tokenizer([r['prompt'] for r in batch],truncation=False,padding=False)['input_ids']
        lengths.update({r['record_id']:len(t) for r,t in zip(batch,tokens)})
    expected={}
    for config in protocol['runs']:
        plan=json.loads((root/(config['name']+'-expected-sampling.json')).read_text())
        if sha(plan)!=protocol['sampling_plan_digests'][config['name']]:raise ValueError('sampling plan substitution')
        ids=plan['record_ids']
        if len(ids)!=8000 or any(r not in by_id for r in ids):raise ValueError('sampling population/work')
        expected[config['name']]={'sample_exposures':len(ids),'unique_training_records':len(set(ids)),'training_tokens':sum(lengths[r] for r in ids),'class_exposures':dict(collections.Counter(by_id[r]['expected_decision'] for r in ids)),'source_exposures':plan['source_exposures']}
    out={'schema':'saal-public-sampler-token-reconstruction-v1','expected':expected,'assumption':'Unchanged native seeded sampler and tokenizer. Matching worker token totals corroborates consistency only; it does not prove a hostile worker sampled these records or used the claimed GPU.'}
    destination=ROOT/'EXPECTED_EXPOSURE.json'
    with destination.open('x') as f:json.dump(out,f,sort_keys=True,indent=2)
    print(json.dumps(expected,indent=2));return out
if __name__=='__main__':reconstruct(sys.argv[1])
