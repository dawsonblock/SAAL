from saal_rsi.checkpoint_selection import metrics
from saal_rsi.learning_dataset import semantic_pilot
from saal_rsi.pretrained_contract import DECISIONS

def test_pairs_exclude_same_label_invariance_controls():
    data=semantic_pilot()['records'];pairs={}
    for r in data:pairs.setdefault(r['cluster_id'],[]).append(r)
    distinct=next(v for v in pairs.values() if len({r['expected_decision'] for r in v})==2)
    same=next(v for v in pairs.values() if len({r['expected_decision'] for r in v})==1 and v[0]['expected_decision']=='REFUSE')
    records=distinct+same
    labels=[distinct[0]['expected_decision']]*2+['REFUSE']*2
    outputs=[{'decision':v,'probabilities':[float(v==d) for d in DECISIONS]} for v in labels]
    result=metrics(records,outputs,'modernbert-seqcls-v1')
    assert result['complete_pair_accuracy']==0 and result['invariance_pair_accuracy']==1

def test_false_refusal_and_missing_clarification():
    records=[next(r for r in semantic_pilot()['records'] if r['expected_decision']==d) for d in DECISIONS]
    labels=['REFUSE','FAST','CHEAP','REFUSE'];outputs=[{'decision':v,'probabilities':[float(v==d) for d in DECISIONS]} for v in labels]
    m=metrics(records,outputs,'modernbert-seqcls-v1');assert m['false_refusal_rate']==.5 and m['missing_clarification_rate']==1 and m['unauthorized_proposals']==1
    assert m['brier_score']==1

def test_generative_decision_diagnostics_preserve_invalid_answers():
    from saal_rsi.generative_diagnostics import decision_diagnostics
    from saal_rsi.task_oracles import oracle
    from saal_rsi.spec import canonical
    records=[{'task_type':'decision','expected_decision':d} for d in ('FAST','ASK','REFUSE')]
    predictions=[canonical(oracle('decision',{'decision':d})).decode() for d in ('REFUSE','FAST','FAST')]
    r=decision_diagnostics(records,predictions)
    assert r['false_refusal_rate']==1 and r['missing_clarification_rate']==1 and r['unsafe_action_proposal_rate']==1
    predictions[1]='malformed';r=decision_diagnostics(records,predictions)
    assert r['invalid_outputs']==1 and r['missing_clarification_rate']==1 and r['confusion']['ASK->INVALID']==1
