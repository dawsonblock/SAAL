import base64
import copy
from dataclasses import asdict,replace
import os
from pathlib import Path
import time
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.spec import BASELINE,canonical,sha
from saal_rsi.discover import discover
from saal_rsi.assurance import create_private_plan,qualify,sign_qualification,verify_attestation
from saal_rsi.registry import Registry,public_hex,verify_receipt
from saal_rsi.promotion import authorize_research_activation,ActivationDenied

@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    directory=tmp_path_factory.mktemp('repairs')
    proposal=discover(development_seeds=[131,132,133],selection_seeds=[141,142,143],worlds=1,episodes=1)
    proposal['candidate']=asdict(BASELINE);proposal['candidate_digest']=BASELINE.identity
    proposal['proposal_digest']=sha({k:v for k,v in proposal.items() if k!='proposal_digest'})
    plan=create_private_plan(seeds=[151,152,153,154,155],cases_per_seed=30)
    custodian=Ed25519PrivateKey.generate();evaluator=Ed25519PrivateKey.generate()
    with Registry(directory/'ledger.db',public_hex(custodian)) as registry:
        receipt=registry.anchor(proposal,plan,custodian)
        report=qualify(proposal,plan,registry=registry,receipt=receipt)
    return proposal,plan,receipt,report,custodian,evaluator


def test_real_report_verifies_with_uppercase_pins(evidence):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    envelope=sign_qualification(report,evaluator)
    assert verify_attestation(envelope,public_hex(evaluator).upper(),proposal=proposal,anchor_public_key=public_hex(custodian).upper())

@pytest.mark.parametrize('field,value',[
    ('baseline_digest','sha256:'+'0'*64),('requires_external_human_approval',False),
    ('paired_gain_mean',1.0),('paired_gain_lower_alpha',1.0)])
def test_even_resigned_inconsistent_reports_fail(evidence,field,value):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    bad=copy.deepcopy(report);bad[field]=value
    assert not verify_attestation(sign_qualification(bad,evaluator),public_hex(evaluator),anchor_public_key=public_hex(custodian))


def test_signature_cannot_turn_failed_statistics_into_pass(evidence):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    bad=copy.deepcopy(report)
    bad['paired_gain_lower_alpha']=1.0;bad['gates']['positive_exploratory_holdout_gain']=True;bad['qualification_passed']=True
    assert not verify_attestation(sign_qualification(bad,evaluator),public_hex(evaluator),anchor_public_key=public_hex(custodian))


def test_signed_behavior_cannot_contradict_proposal_transcript(evidence):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    bad=copy.deepcopy(report);bad['behavior']['episodes'][0]['slot_id']='fast'
    assert not verify_attestation(sign_qualification(bad,evaluator),public_hex(evaluator),anchor_public_key=public_hex(custodian))


def test_signed_sample_counts_and_control_processes_are_checked(evidence):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    for section,field,value in [('candidate_holdout','n',1),('candidate_holdout','abstentions',1)]:
        bad=copy.deepcopy(report);bad[section][field]=value
        assert not verify_attestation(sign_qualification(bad,evaluator),public_hex(evaluator),anchor_public_key=public_hex(custodian))
    bad=copy.deepcopy(report)
    bad['controls']['episodes'][0]['executor_pid']=bad['controls']['episodes'][0]['actor_pid']
    assert not verify_attestation(sign_qualification(bad,evaluator),public_hex(evaluator),anchor_public_key=public_hex(custodian))

@pytest.mark.parametrize('field,value',[
    ('attempt',True),('alpha',.9),('custodian_key_sha256','0'*64),
    ('evaluation_rules',{}),('family','reset'),('anchored_at','no timestamp')])
def test_signed_malformed_receipts_are_rejected(evidence,field,value):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    bad=copy.deepcopy(receipt);bad['payload'][field]=value
    bad['signature_b64']=base64.b64encode(custodian.sign(canonical(bad['payload']))).decode()
    with pytest.raises(ValueError):verify_receipt(bad,public_hex(custodian))


def test_key_casing_cannot_impersonate_independent_human(evidence,tmp_path):
    proposal,plan,receipt,report,custodian,evaluator=evidence
    with pytest.raises(ActivationDenied,match='distinct'):
        authorize_research_activation(proposal,sign_qualification(report,evaluator),{},
            evaluator_public_key=public_hex(evaluator),anchor_public_key=public_hex(custodian),
            human_public_key=public_hex(evaluator).upper(),replay_db=tmp_path/'approvals.db')


def test_registry_pin_case_normalizes_once(tmp_path):
    key=Ed25519PrivateKey.generate()
    with Registry(tmp_path/'db',public_hex(key).upper()) as registry:assert registry.public_key==public_hex(key)
    with Registry(tmp_path/'db',public_hex(key)):pass

@pytest.mark.parametrize('proposal',[{}, {'kind':'UNKNOWN'}, {'kind':'PROPOSE'},
    {'kind':'PROPOSE','slot_id':None}, {'kind':'ASK','question':''},
    {'kind':'ABSTAIN','reason':'none','untrusted_note_seen':1}, {'kind':'TOOL','tool':'system.change_policy','args':[]}])
def test_malformed_actor_responses_fail_closed(proposal):
    from saal_rsi.laboratory import validate_actor_proposal
    with pytest.raises(ValueError):validate_actor_proposal(proposal)


def test_invalid_actor_cannot_receive_appropriate_refusal_credit(monkeypatch):
    import saal_rsi.processes as processes
    from saal_rsi.laboratory import evaluate
    from saal.sim import make_scenario
    class Broken:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def request(self,*args):return {'pid':1,'proposal':{'kind':'UNKNOWN'}}
    monkeypatch.setattr(processes,'Worker',Broken)
    case=make_scenario(1,1);case=replace(case,slots=tuple(replace(s,accessible=False) for s in case.slots))
    with pytest.raises(ValueError,match='proposal kind'):evaluate(BASELINE,[case])


def test_blocked_write_is_included_in_worker_deadline(monkeypatch):
    import saal_rsi.processes as processes
    popen=processes.subprocess.Popen
    def stalled(*args,**kwargs):
        return popen([processes.sys.executable,'-I','-c','import time; time.sleep(120)'],**kwargs)
    monkeypatch.setattr(processes.subprocess,'Popen',stalled)
    with processes.Worker('actor') as worker:
        # Fill the pipe before requesting: timeout must cover a genuinely blocked write.
        for _ in range(1024):
            try:os.write(worker.proc.stdin.fileno(),b'x'*4096)
            except BlockingIOError:break
        else:pytest.fail('pipe did not fill')
        start=time.monotonic()
        with pytest.raises(RuntimeError,match='timeout'):worker.request({'request':'small'},timeout=.1)
        assert time.monotonic()-start<2 and worker.proc.poll() is not None


def test_worker_rejects_false_process_identity(monkeypatch):
    import saal_rsi.processes as processes
    popen=processes.subprocess.Popen
    def spoof(*args,**kwargs):
        code='import sys,json; sys.stdin.readline(); print(json.dumps({"pid":0,"proposal":{"kind":"ABSTAIN","reason":"fake"}}),flush=True)'
        return popen([processes.sys.executable,'-I','-c',code],**kwargs)
    monkeypatch.setattr(processes.subprocess,'Popen',spoof)
    with processes.Worker('actor') as worker:
        with pytest.raises(ValueError,match='process identity'):worker.request({'request':'small'})
        assert worker.proc.poll() is not None


def test_json_io_rejects_symlinks_fifo_and_cleans_failed_outputs(tmp_path):
    from saal_rsi.cli import _read,_write,_output
    original=tmp_path/'input';original.write_text('{"ok":true}')
    linked=tmp_path/'link';linked.symlink_to(original)
    with pytest.raises(OSError):_read(linked)
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    with pytest.raises(ValueError,match='regular'):_read(fifo)
    output=tmp_path/'output'
    with pytest.raises(RuntimeError):
        with _output(output):raise RuntimeError('forced failure')
    assert not output.exists()
    with pytest.raises(ValueError):_write(output,{'nan':float('nan')})
    assert not output.exists()
    _write(output,{'ok':True})
    with pytest.raises(FileExistsError):_write(output,{'ok':False})
    assert _read(output)=={'ok':True}


def test_anchor_output_failure_does_not_spend_alpha(evidence,tmp_path):
    from saal_rsi.cli import main,_write
    from cryptography.hazmat.primitives import serialization
    proposal,plan,receipt,report,custodian,evaluator=evidence
    proposal_path=tmp_path/'proposal';plan_path=tmp_path/'plan';key_path=tmp_path/'key'
    _write(proposal_path,proposal);_write(plan_path,plan)
    key_path.write_bytes(custodian.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption()));key_path.chmod(0o600)
    output=tmp_path/'existing';output.write_text('preserve')
    database=tmp_path/'ledger'
    assert main(['anchor','--proposal',str(proposal_path),'--private-plan',str(plan_path),
        '--custodian-key',str(key_path),'--registry',str(database),'--out',str(output)])==2
    assert output.read_text()=='preserve' and not database.exists()


def test_database_rejects_hardlinks_and_nonregular_files(tmp_path):
    from saal_rsi.storage import private_database
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    with pytest.raises(PermissionError,match='regular'):private_database(fifo)
    first=private_database(tmp_path/'first');os.link(first,tmp_path/'second')
    with pytest.raises(PermissionError,match='hardlink'):private_database(first)
