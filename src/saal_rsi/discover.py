"""Dream-RSI controls *experiment search*, never SAAL runtime or deployment authority."""
from __future__ import annotations
from dataclasses import asdict
from dream_rsi_governed.models import Action, Outcome, ReplayNode, ReplayWorld
from dream_rsi_governed.replay import ReplayEngine
from dream_rsi_governed.policy import FixedBreadthPolicy, RiskAwareUCBPolicy
from .spec import Candidate, BASELINE, sha, initial_candidates
from .laboratory import evaluate, make_cases, quality, summarize
from saal.sim import make_scenario


def replay_worlds(candidates: list[Candidate], seeds: list[int], *, count: int = 5,
                  episodes_per_outcome: int = 12) -> list[ReplayWorld]:
    if not candidates or len({c.identity for c in candidates})!=len(candidates):
        raise ValueError('candidates must be unique')
    if not seeds or len(set(seeds)) != len(seeds):raise ValueError('unique development seeds required')
    if count < 1 or count > 100:raise ValueError('invalid world count')
    worlds=[]
    for wi in range(count):
        nodes=[]
        for c in candidates:
            results=[]
            for batch in range(3):
                seed=seeds[(wi+batch)%len(seeds)]
                # Disjoint index windows across replay worlds, preventing duplicate examples.
                index_start=10000+wi*100+batch*episodes_per_outcome
                cases=[make_scenario(seed,index_start+i,ood=(i%2 == 0)) for i in range(episodes_per_outcome)]
                eps=evaluate(c,cases)
                sm=summarize(eps)
                results.append(Outcome(action_id=c.identity,quality=quality(eps),
                          cost=float(episodes_per_outcome),success=not (sm['violations'] or sm['correction_failures']),
                          failure_class=None if not sm['violations'] else 'constraint',
                          payload={'n':sm['n'],'violations':sm['violations'],
                                   'mean_regret':round(sm['mean_regret'],8),
                                   'correction_failures':sm['correction_failures']}))
            nodes.append(ReplayNode(Action(c.identity,branch=c.name,estimated_cost=episodes_per_outcome),tuple(results)))
        worlds.append(ReplayWorld(f'dev-world-{wi}',f'dev-{wi}',tuple(nodes),budget=max(3,len(candidates)*2),max_parallelism=2,
                         metadata={'source_partition':'development_only'}))
    return worlds


def discover(*, development_seeds: list[int], selection_seeds: list[int],
             worlds: int=3, episodes: int=8, candidates: list[Candidate]|None=None,
             seed: int=29) -> dict:
    if set(development_seeds)&set(selection_seeds):raise ValueError('development/selection seeds overlap')
    if len(development_seeds)<3 or len(selection_seeds)<3:raise ValueError('minimum three seeds per partition')
    candidates=initial_candidates() if candidates is None else candidates
    if BASELINE not in candidates:raise ValueError('incumbent must be included')
    data=replay_worlds(candidates,development_seeds,count=worlds,episodes_per_outcome=episodes)
    # Run both schedulers under identical, per-action replay randomness; no held-out data here.
    results={}
    ucb_observed={}
    for policy in [FixedBreadthPolicy(),RiskAwareUCBPolicy(beta=1.0)]:
        traces=[ReplayEngine(seed=seed).run(w,policy) for w in data]
        observed={}
        for trace in traces:
            for step in trace.steps:
                for out in step.observed:
                    observed.setdefault(out.action_id,[]).append(out.quality)
        if isinstance(policy,RiskAwareUCBPolicy):ucb_observed=observed
        results[policy.policy_id]={'observations':sum(len(v) for v in observed.values()),
               'explored_candidates':len(observed),'mean_best_replay_quality':sum(t.best_quality for t in traces)/len(traces)}
    # Selection is a distinct development-visible partition. Dream proposes the shortlist;
    # no optimization uses the sealed independent evaluator partition.
    # Replay observations determine the top three finalists. Never rerun scheduler.
    ranking=sorted(ucb_observed,key=lambda identity:(-sum(ucb_observed[identity])/len(ucb_observed[identity]),identity))
    shortlist_ids=ranking[:3]
    shortlisted=[c for identity in shortlist_ids for c in candidates if c.identity==identity]
    if BASELINE not in shortlisted:shortlisted.append(BASELINE)
    for metrics in results.values():
        metrics['ucb_replay_ranking']=ranking
        metrics['shortlist_digests']=[c.identity for c in shortlisted]
    scored=[]
    selection_cases=make_cases(selection_seeds,max(episodes*3,15),stress=False)
    for c in shortlisted:
        selected_episodes=evaluate(c,selection_cases)
        s=summarize(selected_episodes)
        scored.append({'candidate':asdict(c),'candidate_digest':c.identity,'development_selection_metrics':s,
                       'selection_quality':quality(selected_episodes)})
    scored.sort(key=lambda x:(-x['selection_quality'], x['candidate_digest']))
    choice=scored[0]
    proposal={'schema':'saal-rsi-proposal-v1','simulation_only':True,
              'candidate':choice['candidate'],'candidate_digest':choice['candidate_digest'],
              'baseline_digest':BASELINE.identity,
              'search':results,'shortlist_count':len(scored),
              'selection_summary':choice['development_selection_metrics'],
              'development_seeds':development_seeds,'selection_seeds':selection_seeds,
              'research_budget':{'worlds':worlds,'episodes_per_outcome':episodes},
              'qualification_status':'NOT_EVALUATED',
              'promotion_status':'PROHIBITED'}
    proposal['proposal_digest']=sha(proposal)
    return proposal
