import pytest
from saal_rsi.continual_learning import retention_metrics
from saal_rsi.experiment_registry import ExperimentRegistry

def test_forgetting_and_transfer():
    r=retention_metrics([[.9,.2,.1],[.7,.8,.3],[.6,.7,.9]],[.1,.1,.1])
    assert r['forgetting_by_task']==pytest.approx([.3,.1,0])
    assert r['backward_transfer']==pytest.approx(-.2)
    assert r['forward_transfer']==pytest.approx(.15)

@pytest.mark.parametrize('matrix,initial',[([[1.2]],[0]),([[1,0]],[0]),([[float('nan')]],[0])])
def test_invalid_retention(matrix,initial):
    with pytest.raises(ValueError):retention_metrics(matrix,initial)
