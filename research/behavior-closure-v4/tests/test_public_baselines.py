import copy,importlib.util,json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('public_baselines',Path(__file__).resolve().parents[1]/'public_baselines.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def dataset():
    records=[]
    for split in ('train','development'):
        for i,label in enumerate(('CHEAP','FAST','ASK','REFUSE')):
            records.append({'record_id':split+str(i),'task_family_id':split,'cluster_id':split+str(i//2),
                            'category':'public','split':split,'prompt':'Choose '+label+'.','expected_decision':label})
    return {'records':records}


def setup_parent(tmp_path,monkeypatch):
    parent=tmp_path/'research/learning-reliability-v2/corpus';parent.mkdir(parents=True)
    (parent/'dataset.json').write_text('{"records":[]}');monkeypatch.setattr(module,'ROOT',tmp_path)


def test_majority_control_never_uses_development_labels(tmp_path,monkeypatch):
    setup_parent(tmp_path,monkeypatch);data=dataset();before=module.report(data)
    changed=copy.deepcopy(data)
    for row in changed['records']:
        if row['split']=='development':row['expected_decision']='REFUSE'
    after=module.report(changed)
    assert before['majority_label']==after['majority_label']=='CHEAP'
    assert not before['measurements']['development']['all']['training_majority']['screen']['passes_public_screen']


def test_missing_training_class_cannot_create_admitted_control(tmp_path,monkeypatch):
    setup_parent(tmp_path,monkeypatch);data=dataset();data['records']=[r for r in data['records'] if r['split']!='train' or r['expected_decision']!='ASK']
    with pytest.raises(ValueError):module.report(data)
