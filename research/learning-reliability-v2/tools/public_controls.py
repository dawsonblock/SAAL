"""Retrospective fixed unigram control; fits training only, never selects a model."""
from collections import Counter
import math,re
from saal_rsi.pretrained_contract import DECISIONS

def tokens(text):
    return ["NUMBER" if t.isdigit() else t for t in re.findall(r"[a-z]+|[0-9]+",text.lower())]

class UnigramControl:
    def __init__(self,records):
        if not records or any(r['split']!='train' for r in records):raise ValueError('training-only control')
        self.counts={d:Counter() for d in DECISIONS};self.classes=Counter();self.vocabulary=set()
        for r in records:
            d=r['expected_decision']
            if d not in self.counts:raise ValueError('unknown label')
            ts=tokens(r['prompt']);self.counts[d].update(ts);self.classes[d]+=1;self.vocabulary.update(ts)
        if set(self.classes)!=set(DECISIONS):raise ValueError('missing class')
        self.totals={d:sum(c.values()) for d,c in self.counts.items()};self.n=sum(self.classes.values())
    def predict(self,prompt):
        ts=Counter(t for t in tokens(prompt) if t in self.vocabulary)
        scores=[math.log(self.classes[d]/self.n)+sum(n*math.log((self.counts[d][t]+1)/(self.totals[d]+len(self.vocabulary))) for t,n in sorted(ts.items())) for d in DECISIONS]
        m=max(scores);p=[math.exp(s-m) for s in scores];total=sum(p);p=[v/total for v in p]
        return {'decision':DECISIONS[max(range(len(scores)),key=scores.__getitem__)],'probabilities':p}
