import json
import shutil
import sys
import pytest
from saal_rsi.restricted_execution import restricted_process


@pytest.mark.skipif(sys.platform!='darwin' or shutil.which('sandbox-exec')is None,reason='macOS process restriction is platform-specific')
def test_real_child_cannot_read_authority_secrets_or_inherit_credentials(tmp_path,monkeypatch):
    private=tmp_path/'custodian';private.mkdir();(private/'holdout.json').write_text('CONFIDENTIAL_HOLDOUT_SENTINEL');(private/'key').write_text('SECRET_KEY_SENTINEL')
    output=tmp_path/'worker';output.mkdir();monkeypatch.setenv('CUSTODIAN_PRIVATE_KEY','must_not_inherit')
    code='''import json,os,socket
from pathlib import Path
denied=[]
for name in ('holdout.json','key'):
 try:Path(os.environ.get('NO_SUCH_VAR','') or PRIVATE_ROOT,name).read_text();denied.append(False)
 except PermissionError:denied.append(True)
try:socket.create_connection(('127.0.0.1',9),timeout=1);network_denied=False
except PermissionError:network_denied=True
except OSError:network_denied=False
Path('permitted-output.json').write_text(json.dumps({'denied':denied,'network_denied':network_denied,'credential_absent':'CUSTODIAN_PRIVATE_KEY' not in os.environ}))
'''
    code='PRIVATE_ROOT='+repr(str(private))+'\n'+code
    p=restricted_process([sys.executable,'-c',code],output,[private])
    assert p.returncode==0,p.stderr
    result=json.loads((output/'permitted-output.json').read_text())
    assert result=={'denied':[True,True],'network_denied':True,'credential_absent':True}


def test_unsupported_isolation_has_no_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr('saal_rsi.restricted_execution.shutil.which',lambda _:None)
    with pytest.raises(PermissionError):restricted_process([sys.executable,'-c','print(1)'],tmp_path,[tmp_path/'private'])
