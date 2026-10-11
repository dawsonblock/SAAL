import sys,json
import pytest
from saal_rsi.pretrained_cli import main

def test_cli_qwen_defaults_to_separately_tested_fp32(monkeypatch,tmp_path):
    out=tmp_path/'spec.json';monkeypatch.setattr(sys,'argv',['saal-pretrained','model-spec','--repository','Qwen/Qwen3.5-0.8B','--out',str(out)])
    main();assert json.loads(out.read_text())['precision']=='float32'
    with pytest.raises(FileExistsError):main()

def test_cli_unknown_precision_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['saal-pretrained','model-spec','--repository','Qwen/Qwen3.5-0.8B','--precision','int4','--out',str(tmp_path/'spec.json')])
    with pytest.raises(SystemExit):main()
