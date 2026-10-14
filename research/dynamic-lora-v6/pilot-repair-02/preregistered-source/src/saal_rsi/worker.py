"""Fresh actor/executor interpreter. Actor gets public views only; no inherited secrets."""
import os
from pathlib import Path
import secrets
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saal_rsi.spec import Candidate, canonical, parse_unique_json
from saal_rsi.processes import MAX_MESSAGE


def run(role):
    if role == 'actor':
        from saal_rsi.actor import fixture_response
    elif role == 'executor':
        from saal.authority import Authority, SimulatedExecutor
        from saal.types import Action, Delegation
        authority = Authority(secrets.token_bytes(32), secrets.token_bytes(32))
        executor = SimulatedExecutor(authority)
    else: raise ValueError('unknown role')
    for line in iter(lambda: sys.stdin.buffer.readline(MAX_MESSAGE+1), b''):
        try:
            if len(line)>MAX_MESSAGE or not line.endswith(b'\n'): raise ValueError('invalid IPC frame')
            req=parse_unique_json(line)
            if role=='actor':
                if set(req)!={'candidate','view','answer','fixture'}: raise ValueError('actor IPC shape')
                if req['fixture'] not in {'safe','injection','proxy','abstain','wrong-answer','eval-sensitive','unsafe'}:
                    raise ValueError('unknown fixture')
                result=fixture_response(req['fixture'], Candidate.parse(req['candidate']),req['view'],req['answer'])
                result={'proposal':result,'pid':os.getpid()}
            else:
                operation=req.get('op')
                if operation=='authorize':
                    if set(req)!={'op','action'}:raise ValueError('executor shape')
                    action=Action(**req['action'])
                    # Delegations are installed by executor code, never actor-supplied.
                    if action.principal!='simulation': raise ValueError('invalid principal')
                    delegation=Delegation('simulation',action.task_id,'nonbinding proposal',
                        ('schedule.propose',),2_000_000_000,max_actions=1)
                    decision=authority.decide(action,delegation,now=1_800_000_000)
                    result={'status':decision.disposition,'token':decision.token,'pid':os.getpid()}
                elif operation=='execute':
                    if set(req)!={'op','action','token','allowed_slots'}:raise ValueError('executor shape')
                    action=Action(**req['action'])
                    if action.tool!='schedule.propose' or action.args.get('slot_id') not in req['allowed_slots']:
                        result={'executed':False,'pid':os.getpid()}
                    else:
                        try:
                            event=executor.execute(action,req['token'],now=1_800_000_001)
                            result={'executed':True,'event':event,'pid':os.getpid()}
                        except PermissionError:
                            result={'executed':False,'pid':os.getpid()}
                else:raise ValueError('unknown executor operation')
            sys.stdout.buffer.write(canonical(result)+b'\n');sys.stdout.buffer.flush()
        except Exception as exc:
            sys.stdout.buffer.write(canonical({'error':type(exc).__name__})+b'\n');sys.stdout.buffer.flush()
    if role=='executor': authority.close()

if __name__=='__main__':run(sys.argv[1])
