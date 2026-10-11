import json,hashlib
from pathlib import Path
import pytest
from verify import verify

def fixture(root):
 p=root/'data.json';p.write_text('{"instruction":"Untrusted task content"}');source={'repository':'source','revision':'a'*40,'exposure':'PUBLIC','qualification_allowed':False,'saal_labels_present':False,'files':[{'path':'data.json','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]};m={'schema':'saal-public-data-acquisition-v1','qualification_status':'NOT_EVALUATED','saal_training_admitted':False,'sources':[source],'model_references':[]};(root/'ACQUISITION.json').write_text(json.dumps(m));return m

def test_verified_public_source_grants_no_training_or_qualification(tmp_path):
 fixture(tmp_path);r=verify(tmp_path);assert r['source_files_verified']==1 and r['saal_training_admitted']is False

def test_changed_source_is_rejected(tmp_path):
 fixture(tmp_path);(tmp_path/'data.json').write_text('changed');
 with pytest.raises(ValueError,match='substitution'):verify(tmp_path)

@pytest.mark.parametrize('change',['qualification','automatic_saal_labels','floating_revision','training','path_escape','duplicate'])
def test_invalid_provenance_or_authority_rejected(tmp_path,change):
 m=fixture(tmp_path);s=m['sources'][0]
 if change=='qualification':s['qualification_allowed']=True
 elif change=='automatic_saal_labels':s['saal_labels_present']=True
 elif change=='floating_revision':s['revision']='main'
 elif change=='training':m['saal_training_admitted']=True
 elif change=='path_escape':s['files'][0]['path']='../outside'
 else:s['files'].append(s['files'][0].copy())
 (tmp_path/'ACQUISITION.json').write_text(json.dumps(m))
 with pytest.raises(ValueError):verify(tmp_path)

def test_symlink_is_rejected(tmp_path):
 fixture(tmp_path);(tmp_path/'original.json').write_text((tmp_path/'data.json').read_text());(tmp_path/'data.json').unlink();(tmp_path/'data.json').symlink_to(tmp_path/'original.json')
 with pytest.raises(ValueError,match='inventory'):verify(tmp_path)
