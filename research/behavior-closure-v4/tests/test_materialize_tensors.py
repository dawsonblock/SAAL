import hashlib,json
from pathlib import Path
import pytest
import importlib.util
spec=importlib.util.spec_from_file_location('materializer',Path(__file__).resolve().parents[1]/'materialize_tensors.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_all_bytes_are_present_and_restore_is_idempotent(tmp_path):
    data=b'complete tensor test bytes';h=hashlib.sha256(data).hexdigest()
    payload='tensor_payloads/'+h+'.safetensors';(tmp_path/'tensor_payloads').mkdir();(tmp_path/payload).write_bytes(data)
    files={p:{'sha256':h,'size':len(data),'payload':payload} for p in ['model/initial.safetensors','model/best.safetensors']}
    (tmp_path/'TENSOR_PAYLOADS.json').write_text(json.dumps({'schema':'saal-physical-tensor-payloads-v1','files':files}))
    assert module.materialize(tmp_path)==2 and module.materialize(tmp_path)==2
    assert (tmp_path/'model/initial.safetensors').read_bytes()==data
    assert (tmp_path/'model/initial.safetensors').stat().st_ino==(tmp_path/'model/best.safetensors').stat().st_ino


@pytest.mark.parametrize('fault',['missing','corrupted','escape','existing_substitution'])
def test_missing_bytes_and_substitution_rejected(tmp_path,fault):
    data=b'complete bytes';h=hashlib.sha256(data).hexdigest();(tmp_path/'tensor_payloads').mkdir()
    payload='tensor_payloads/'+h+'.safetensors';p=tmp_path/payload;p.write_bytes(data)
    name='model/initial.safetensors'
    if fault=='missing':p.unlink()
    elif fault=='corrupted':p.write_bytes(b'corrupt bytes!')
    elif fault=='escape':name='../initial.safetensors'
    else:(tmp_path/'model').mkdir();(tmp_path/name).write_bytes(b'substituted')
    (tmp_path/'TENSOR_PAYLOADS.json').write_text(json.dumps({'schema':'saal-physical-tensor-payloads-v1','files':{name:{'sha256':h,'size':len(data),'payload':payload}}}))
    with pytest.raises((ValueError,FileExistsError)):module.materialize(tmp_path)
