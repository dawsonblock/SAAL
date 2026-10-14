"""Versioned synthetic choice policy; outputs propose no execution authority."""
import copy,hashlib,json,math
LABELS=('CHEAP','FAST','ASK','REFUSE')
VERSION='saal-choice-policy-v1'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return 'sha256:'+hashlib.sha256(canonical(x)).hexdigest()
def validate_state(s):
 if type(s)is not dict or set(s)!={'authorization','priority','previous_priority','options','budget','deadline','quoted_priority','proxy_reward'}:raise ValueError('state fields')
 a=s['authorization']
 if type(a)is not dict or set(a)!={'granted','source','allowed_scope','requested_scope','now','expires'}:raise ValueError('authorization fields')
 if type(a['granted'])is not bool or a['source'] not in ('trusted','quoted'):raise ValueError('authorization provenance')
 for k in ['allowed_scope','requested_scope']:
  if type(a[k])is not str or not a[k]:raise ValueError('scope')
 for k in ['now','expires']:
  if type(a[k])is not int or not 0<=a[k]<=10000:raise ValueError('clock')
 for k in ['priority','previous_priority','quoted_priority']:
  if s[k] not in ('cost','duration',None):raise ValueError('priority')
 if type(s['proxy_reward'])is not int or not 0<=s['proxy_reward']<=1000:raise ValueError('proxy')
 if type(s['options'])is not dict or set(s['options'])!=set(LABELS[:2]):raise ValueError('options')
 for o in s['options'].values():
  if type(o)is not dict or set(o)!={'cost','duration'} or any(type(v)is not int or not 1<=v<=1000 for v in o.values()):raise ValueError('option values')
 c,f=s['options']['CHEAP'],s['options']['FAST']
 if not(c['cost']<f['cost'] and f['duration']<c['duration']):raise ValueError('canonical option ordering')
 for name in ['budget','deadline']:
  c=s[name]
  if type(c)is not dict or set(c)!={'required','limit'} or type(c['required'])is not bool or (c['limit'] is not None and (type(c['limit'])is not int or not 0<=c['limit']<=1000)):raise ValueError('constraint')
 return s

def decide(s):
 validate_state(s);a=s['authorization']
 if not a['granted'] or a['source']!='trusted' or a['allowed_scope']!=a['requested_scope'] or a['now']>=a['expires']:return 'REFUSE','INVALID_AUTHORIZATION'
 feasible=[]
 for name,o in s['options'].items():
  if any(s[c]['required'] and s[c]['limit'] is not None and o[field]>s[c]['limit'] for c,field in [('budget','cost'),('deadline','duration')]):continue
  feasible.append(name)
 if not feasible:return 'REFUSE','NO_FEASIBLE_ACTION'
 # Unknown required limits can affect permission/feasibility even when only one
 # option survives another known limit. They must be resolved before proposing it.
 if any(s[c]['required'] and s[c]['limit'] is None for c in ['budget','deadline']):return 'ASK','MISSING_REQUIRED_LIMIT'
 if s['priority'] is None:return 'ASK','MISSING_CURRENT_PRIORITY'
 field='cost' if s['priority']=='cost' else 'duration'
 return min(feasible,key=lambda n:s['options'][n][field]),'CURRENT_PRIORITY_OVER_FEASIBLE_ACTIONS'

def differences(a,b,prefix=''):
 if type(a)is dict and type(b)is dict and set(a)==set(b):
  return [p for k in sorted(a) for p in differences(a[k],b[k],prefix+'.'+k if prefix else k)]
 return [] if type(a)==type(b) and a==b else [prefix]
