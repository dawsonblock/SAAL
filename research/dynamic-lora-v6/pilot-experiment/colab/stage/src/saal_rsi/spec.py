"""Closed candidate vocabulary. This integration never accepts executable policy code."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(data: object) -> str:
    return 'sha256:' + hashlib.sha256(canonical(data)).hexdigest()


@dataclass(frozen=True)
class Candidate:
    name: str
    ask_cost: float = .2
    clarify: bool = True
    infer: bool = True
    controlled: bool = True
    model_kind: str = 'frozen-rules-v1'

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ValueError('candidate name required')
        if type(self.ask_cost) is not float or not math.isfinite(self.ask_cost) or not 0 <= self.ask_cost <= 4:
            raise ValueError('ask_cost must be a finite floating point number in [0,4]')
        if not all(type(x) is bool for x in (self.infer, self.clarify, self.controlled)):
            raise ValueError('invalid Boolean controls')
        if self.controlled is not True or self.infer is not True:
            raise ValueError('actor cannot disable external authority or goal inference')
        if self.model_kind != 'frozen-rules-v1':
            raise ValueError('unknown or mutable backend')

    @property
    def identity(self) -> str:
        return sha(asdict(self))

    @classmethod
    def parse(cls, record: dict) -> 'Candidate':
        if type(record) is not dict or set(record) != set(cls.__dataclass_fields__):
            raise ValueError('candidate schema mismatch')
        return cls(**record)


BASELINE = Candidate(name='incumbent', ask_cost=.2)


def initial_candidates() -> list[Candidate]:
    return [BASELINE] + [Candidate(name=f'ask-{v:g}', ask_cost=float(v)) for v in [0, .05, .1, .3, .5, .8, 1.2, 2.0, 4.0]]


def parse_unique_json(data: bytes) -> dict:
    def strict(pairs):
        out = {}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    result = json.loads(data.decode('utf-8'), object_pairs_hook=strict,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON number')))
    if not isinstance(result,dict): raise ValueError('expected JSON object')
    return result
