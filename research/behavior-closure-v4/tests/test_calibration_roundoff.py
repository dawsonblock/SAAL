import copy,math,json
from pathlib import Path
import pytest
from saal_rsi.epoch_artifacts_v2 import metrics_equivalent,validate_bundle
from saal_rsi.model_contract import verify


def test_one_ulp_calibration_roundoff_has_no_effect_on_behavioral_rates():
    a={'accuracy':.8,'unauthorized_proposals':0,'brier_score':.6398315521858382,
       'negative_log_likelihood_by_class':{'CHEAP':.4},'nll_probability_floor':1e-12}
    b=copy.deepcopy(a);b['brier_score']=math.nextafter(a['brier_score'],math.inf)
    assert metrics_equivalent(a,b)
    b['accuracy']=math.nextafter(a['accuracy'],math.inf)
    assert not metrics_equivalent(a,b)


@pytest.mark.parametrize('fault',['large_calibration','critical_count','boolean_count','nan','class_nll','extra_field'])
def test_forged_metrics_remain_rejected(fault):
    a={'accuracy':.8,'unauthorized_proposals':0,'brier_score':.6,'negative_log_likelihood_by_class':{'FAST':.4}}
    b=copy.deepcopy(a)
    if fault=='large_calibration':b['brier_score']+=1e-12
    elif fault=='critical_count':b['unauthorized_proposals']=1
    elif fault=='boolean_count':b['unauthorized_proposals']=False
    elif fault=='nan':b['brier_score']=float('nan')
    elif fault=='class_nll':b['negative_log_likelihood_by_class']['FAST']+=1e-12
    else:b['unregistered']=True
    assert not metrics_equivalent(a,b)


def test_probabilities_are_not_a_calibration_roundoff_exception():
    a={'probabilities':[.6,.4],'worst_family_accuracy':.8}
    b=copy.deepcopy(a);b['probabilities'][0]=math.nextafter(.6,math.inf)
    assert not metrics_equivalent(a,b)
