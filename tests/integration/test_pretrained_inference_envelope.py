"""Exercise real exporter/parser plumbing with deterministic model stubs."""
import json
from types import SimpleNamespace
import pytest
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.pretrained_inference import main, prediction_envelope
from saal_rsi.spec import canonical


def test_inference_child_export_matches_parent_reader(tmp_path,monkeypatch):
    import saal_rsi.learning_backend as backend
    import saal_rsi.pretrained_registry as registry
    import saal_rsi.pretrained_inference as worker
    output=[{'decision':'FAST','probabilities':[0.,1.,0.,0.]}]
    model=SimpleNamespace(model=SimpleNamespace(to=lambda device:None),
        load_pretrained=lambda: model, restore_checkpoint=lambda path:None,
        evaluate=lambda rows:output)
    monkeypatch.setattr(backend,'create_backend',lambda *args:model)
    monkeypatch.setattr(backend,'verify_dependencies',lambda:{})
    monkeypatch.setattr(registry,'validate_snapshot',lambda *args:None)
    request={'model_spec':model_spec('answerdotai/ModernBERT-base',method='head',backend='local',precision='float32',budget='serialization_fixture'),
        'seed':47,'checkpoint':str(tmp_path/'fixture.safetensors'),'cache':str(tmp_path),
        'prompts':['Synthetic bounded request; choose speed with permission.']}
    source=tmp_path/'input.json';target=tmp_path/'out.json';source.write_bytes(canonical(request))
    monkeypatch.setattr(worker.sys,'argv',['worker',str(source),str(target)])
    main()
    assert prediction_envelope(target.read_bytes())==output


@pytest.mark.parametrize('raw',[b'[]',b'{"predictions":[],"approved":true}',b'{"predictions":{},"predictions":[]}'])
def test_malformed_export_never_becomes_predictions(raw):
    with pytest.raises(ValueError):prediction_envelope(raw)
