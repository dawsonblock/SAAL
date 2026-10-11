"""v3 threat-oriented tests for identity spoofing, replay, and sealed studies."""
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from saal.authority import Authority
from saal.canonical import digest, sign
from saal.evaluation import run_study
from saal.gate_server import GateServer, load_peer_bindings
from saal.prereg import private_prereg, receipt
from saal.qualification import qualify
from saal.replay import audit_replay
from saal.train import fit_intent_model
from saal.types import Delegation


@unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'Unix socket required')
class PeerIdentityTests(unittest.TestCase):
    def test_peer_scope_enforced_even_for_preinstalled_delegations(self):
        with tempfile.TemporaryDirectory() as directory:
            authority = Authority(b'A'*32, b'B'*32)
            delegated = [
                Delegation('alice', 'one', 'example', ('schedule.read',), 2_000_000_000),
                Delegation('bob', 'two', 'example', ('schedule.read',), 2_000_000_000),
            ]
            path = str(Path(directory)/'authority.sock')
            with GateServer(path, authority, delegated, peer_bindings={
                    os.getuid(): frozenset({('alice', 'one')})}) as server:
                background = threading.Thread(target=server.serve_forever, daemon=True)
                background.start()
                try:
                    def send(principal, task, raw=None):
                        action = {'action': {'tool': 'schedule.read',
                                'args': {'calendar_id': 'test'},
                                'principal': principal, 'task_id': task}}
                        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                            sock.connect(path)
                            sock.sendall((raw or json.dumps(action)) .encode() + b'\n')
                            return json.loads(sock.makefile('rb').readline())
                    self.assertEqual(send('alice', 'one')['status'], 'ALLOW')
                    denied = send('bob', 'two')
                    self.assertEqual(denied['status'], 'DENY')
                    self.assertIn('peer identity', denied['reasons'][0])
                    bad_json = ('{"action":{"tool":"schedule.read",'
                                '"args":{"calendar_id":"test"},'
                                '"principal":"alice","task_id":"one",'
                                '"task_id":"one"}}')
                    self.assertEqual(send('alice', 'one', bad_json)['status'], 'DENY')
                    with patch('saal.gate_server._peer_uid', return_value=999999):
                        self.assertEqual(send('alice', 'one')['status'], 'DENY')
                finally:
                    server.shutdown()
                    background.join(3)
            authority.close()

    def test_server_requires_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            authority = Authority(b'A'*32, b'B'*32)
            try:
                with self.assertRaises(ValueError):
                    GateServer(str(Path(directory)/'auth.sock'), authority, [], peer_bindings={})
            finally:
                authority.close()

    def test_trusted_peer_mapping_loads_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'peers.json'
            path.write_text(json.dumps({str(os.getuid()): [
                {'principal':'alice','task_id':'one'}]}))
            os.chmod(path, 0o600)
            mapping = load_peer_bindings(str(path))
            self.assertIn(('alice', 'one'), mapping[os.getuid()])
            path.write_text('{"123":[{"principal":"a","task_id":"x"}],'
                            '"123":[{"principal":"b","task_id":"x"}]}')
            with self.assertRaises(ValueError):
                load_peer_bindings(str(path))
            os.chmod(path, 0o644)
            with self.assertRaises(PermissionError):
                load_peer_bindings(str(path))


class QualificationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_intent_model(seed=1201, epochs=1, per_class=5)
        cls.eval_key = b'E'*32
        cls.prereg = {'schema_version':1, 'seeds':[7301,7303,7307],
                      'episodes_per_seed': 4, 'min_episodes':12,
                      'candidate_arm':'F_offline_model_controlled',
                      'baseline_arm':'E_frozen_controlled',
                      'min_improvement_lower_bound': -99.0,
                      'max_critical_event_rate_upper_bound': 0.99,
                      'require_replay':True}

    def fresh_report(self):
        return run_study(self.model, seeds=self.prereg['seeds'], episodes_per_seed=4,
                         eval_key=self.eval_key, prereg=self.prereg)

    def test_valid_replay_not_automatic_promotion(self):
        report = self.fresh_report()
        self.assertTrue(audit_replay(report, self.prereg, self.model, self.eval_key)['valid'])
        no_model = qualify(report, self.eval_key, self.prereg)
        self.assertEqual(no_model['verdict'], 'REJECT')
        self.assertTrue(any('replay requires' in s for s in no_model['reasons']))
        yes_model = qualify(report, self.eval_key, self.prereg, model=self.model)
        self.assertTrue(yes_model['replay_audit']['valid'])
        self.assertTrue(yes_model['caution'])

    def test_forged_signed_outcome_detected_by_replay(self):
        report = self.fresh_report()
        arm = report['report']['results']['F_offline_model_controlled']
        arm['episodes'][0]['true_utility'] += 0.75
        # Compromised evaluator recomputes aggregate to cover its falsified record.
        import statistics
        arm['metrics']['mean_true_utility'] = statistics.fmean(
            e['true_utility'] for e in arm['episodes'])
        report['report_digest'] = digest(report['report'])
        report['signature'] = sign(report['report'], self.eval_key)
        self.assertFalse(audit_replay(report, self.prereg, self.model, self.eval_key)['valid'])
        qualification = qualify(report, self.eval_key, self.prereg, model=self.model)
        self.assertEqual(qualification['verdict'], 'REJECT')
        self.assertTrue(any('replay mismatch' in x for x in qualification['reasons']))

    def test_forged_aggregate_detected_even_if_signed(self):
        report = self.fresh_report()
        arm = report['report']['results']['F_offline_model_controlled']
        arm['metrics']['blocked_injections'] = 888
        report['report_digest'] = digest(report['report'])
        report['signature'] = sign(report['report'], self.eval_key)
        q = qualify(report, self.eval_key, self.prereg, model=self.model)
        self.assertEqual(q['verdict'], 'REJECT')
        self.assertIn('inconsistent signed metric: blocked_injections', q['reasons'])

    def test_private_prereg_generation(self):
        a = private_prereg(12,30)
        b = private_prereg(12,30)
        self.assertEqual(len(set(a['seeds'])),12)
        self.assertNotEqual(a['seeds'], b['seeds'])
        self.assertNotIn(1201, a['seeds'])
        self.assertTrue(a['require_replay'])
        self.assertEqual(receipt(a)['prereg_digest'], digest(a))
        with self.assertRaises(ValueError):
            private_prereg(3,2)


if __name__ == '__main__':
    unittest.main()

class ActionBudgetV3Tests(unittest.TestCase):
    def test_zero_cost_actions_exhaust_the_count_budget_and_survive_restart(self):
        from saal.types import Action, Delegation
        from saal.authority import SimulatedExecutor
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory)/'spent.sqlite')
            key, human = b'A'*32, b'B'*32
            delegation = Delegation('alice','task','example',('schedule.propose',),
                                    2_000_000_000, max_actions=2)
            actor_action = Action('schedule.propose', {'slot_id':'B','cost_cents':0},
                                  'alice','task')
            a = Authority(key, human, replay_db=db)
            first = a.decide(actor_action, delegation, now=1_800_000_000)
            self.assertEqual(first.disposition, 'ALLOW')
            self.assertEqual(first.token['claims']['version'], 3)
            tampered = json.loads(json.dumps(first.token))
            tampered['claims']['action_limit'] = 1_000_000
            self.assertFalse(a.consume(tampered, actor_action, now=1_800_000_001))
            SimulatedExecutor(a).execute(actor_action, first.token, now=1_800_000_001)
            second = a.decide(actor_action, delegation, now=1_800_000_002)
            self.assertEqual(second.disposition, 'ALLOW')
            third = a.decide(actor_action, delegation, now=1_800_000_003)
            self.assertEqual(third.disposition, 'DENY')
            self.assertIn('action limit', third.reasons[0])
            a.close()
            reopened = Authority(key, human, replay_db=db)
            try:
                blocked = reopened.decide(actor_action, delegation, now=1_800_000_004)
                self.assertEqual(blocked.disposition, 'DENY')
                self.assertTrue(reopened.consume(second.token, actor_action, now=1_800_000_005))
                self.assertEqual(reopened.decide(actor_action, delegation, now=1_800_000_065).disposition,
                                 'DENY')
            finally:
                reopened.close()

    def test_expired_unused_count_reservation_is_released(self):
        from saal.types import Action, Delegation
        a=Authority(b'A'*32,b'B'*32)
        try:
            d=Delegation('alice','task','example',('schedule.propose',),
                         2_000_000_000,max_actions=1)
            act=Action('schedule.propose',{'slot_id':'A','cost_cents':0},'alice','task')
            self.assertEqual(a.decide(act,d,now=1_800_000_000).disposition,'ALLOW')
            self.assertEqual(a.decide(act,d,now=1_800_000_001).disposition,'DENY')
            self.assertEqual(a.decide(act,d,now=1_800_000_062).disposition,'ALLOW')
        finally:
            a.close()

    def test_old_replay_db_fails_closed(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            db=Path(directory)/'old.sqlite'
            con=sqlite3.connect(db)
            con.execute('CREATE TABLE tokens(nonce TEXT PRIMARY KEY, used INTEGER NOT NULL)')
            con.commit();con.close()
            with self.assertRaises(RuntimeError):
                Authority(b'A'*32,b'B'*32,replay_db=db)
