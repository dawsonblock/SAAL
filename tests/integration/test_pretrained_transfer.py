import hashlib
import pytest
from saal_rsi.pretrained_backend import validate_transfer,assemble_transfer,CHUNK_BYTES

def manifest(raw):return {'schema':'saal-artifact-transfer-v1','job_digest':'sha256:'+'a'*64,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'chunks':[{'name':'part-000.bin','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}

def test_verified_chunk_assembly(tmp_path):
    data=b'actual checkpoint bytes';(tmp_path/'part-000.bin').write_bytes(data);m=manifest(data);assemble_transfer(tmp_path,m,tmp_path/'result.zip');assert (tmp_path/'result.zip').read_bytes()==data

def test_corruption_fails_closed(tmp_path):
    m=manifest(b'expected');(tmp_path/'part-000.bin').write_bytes(b'attacker')
    with pytest.raises(ValueError):assemble_transfer(tmp_path,m,tmp_path/'result.zip')

@pytest.mark.parametrize('mutation',['traversal','order','size','count','sum'])
def test_untrusted_transfer_manifest(mutation):
    m=manifest(b'abc')
    if mutation=='traversal':m['chunks'][0]['name']='../private.key'
    elif mutation=='order':m['chunks'][0]['name']='part-003.bin'
    elif mutation=='size':m['chunks'][0]['size']=CHUNK_BYTES+1
    elif mutation=='count':m['chunks']*=66
    else:m['size']=4
    with pytest.raises(ValueError):validate_transfer(m)

def test_remote_python_failure_is_not_hidden_by_cli_success(tmp_path):
    import sys
    from saal_rsi.pretrained_backend import run_remote_stage
    code="import json;print(json.dumps({'monitor':'worker_failure','stage':'training','returncode':1}))"
    with pytest.raises(RuntimeError,match='Remote worker failed'):
        run_remote_stage([sys.executable,'-c',code],tmp_path/'failed.log',5)
    assert 'worker_failure' in (tmp_path/'failed.log').read_text()
