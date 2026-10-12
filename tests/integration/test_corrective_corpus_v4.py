import copy
import json
from pathlib import Path
import pytest
from saal_rsi.corrective_corpus import build, validate, render
from saal_rsi.learning_dataset import semantic_pilot
from saal_rsi.choice_policy_v1 import decide, differences


@pytest.fixture(scope='module')
def pilot():
    parent=semantic_pilot();dataset,audit=build(parent,2)
    return parent,dataset,audit


def test_labels_pairs_and_old_records_are_preserved(pilot):
    parent,dataset,audit=pilot;result=validate(dataset,audit,parent)
    assert result['new_records']==288 and result['new_pairs']==144
    assert dataset['records'][:len(parent['records'])]==parent['records']
    assert result['human_reviews']==0 and result['independent_population'] is False
    states={r['record_id']:r for r in audit['records']}
    for r in dataset['records'][len(parent['records']):]:
        s=states[r['record_id']]
        assert decide(s['state'])[0]==r['expected_decision']
        assert render(s['state'],s['recipe'],s['number'],s['quoted'])==r['prompt']
        if r['parent_record_id']:
            assert differences(states[r['parent_record_id']]['state'],s['state'])==[s['changed_field']]


@pytest.mark.parametrize('field', ['prompt','expected_decision','split','parent_record_id'])
def test_semantic_and_lineage_tampering_rejected(pilot,field):
    parent,dataset,audit=pilot;changed=copy.deepcopy(dataset)
    changed['records'][-1][field]='tampered'
    with pytest.raises(ValueError):validate(changed,audit,parent)


def test_prompt_audit_not_replaced_by_oracle_labels(pilot):
    parent,dataset,audit=pilot;wrong=copy.deepcopy(audit)
    wrong['records'][-1]['state']['authorization']['granted']=False
    with pytest.raises(ValueError):validate(dataset,wrong,parent)


def test_rejected_qualification_cli_writes_private_incident(tmp_path):
    import os,subprocess,sys
    out=tmp_path/'incident.json'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]/'src'))
    result=subprocess.run([sys.executable,'-m','saal_rsi.pretrained_qualification_cli','freeze',
        '--bundle',str(tmp_path/'missing'),'--controller-pin','00'*32,'--out',str(out)],env=env,capture_output=True)
    assert result.returncode != 0
    assert json.loads(out.read_text())['status']=='FAILED'
    assert out.stat().st_mode & 0o077 == 0
    previous=out.read_bytes()
    subprocess.run([sys.executable,'-m','saal_rsi.pretrained_qualification_cli','freeze',
        '--bundle',str(tmp_path/'missing'),'--controller-pin','00'*32,'--out',str(out)],env=env,capture_output=True)
    assert out.read_bytes()==previous
