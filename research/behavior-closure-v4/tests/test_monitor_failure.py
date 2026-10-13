import importlib.util,json,time
from pathlib import Path


def test_terminal_campaign_event_does_not_hide_failed_lifecycle(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('behavior_monitor',Path(__file__).resolve().parents[1]/'monitor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'PROJECT',tmp_path)
    folder=tmp_path/'experiment/continuation-01/colab';folder.mkdir(parents=True)
    (folder/'monitor-state.json').write_text(json.dumps({'session':'test','phase':'FAILED','heartbeat_epoch':time.time(),
        'terminal':True,'cleanup_verified':True,'error':{'type':'TimeoutError','detail':'export cleanup interrupted'}}))
    (folder/'4-campaign.log').write_text(json.dumps({'monitor':'campaign_progress','completed_runs':0,'total_runs':7,'seed':None,'mode':'TERMINAL'})+'\n')
    result=module.snapshot()
    assert result['stages']['continuation']['stage']=='FAILED'
    assert result['stages']['continuation']['cleanup_verified'] is True
    assert result['completed_new_model_runs']==0
    assert result['external_deployment_allowed'] is False
