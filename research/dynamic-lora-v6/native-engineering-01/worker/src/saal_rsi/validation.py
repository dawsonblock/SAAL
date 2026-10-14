"""Shared, typed protocol constraints and canonical role-key identity."""
import math


def key_bytes(value):
    if type(value) is not str or len(value)!=64:
        raise ValueError('public key must be 64 hexadecimal characters')
    try:raw=bytes.fromhex(value)
    except ValueError as exc:raise ValueError('invalid public key encoding') from exc
    if len(raw)!=32:raise ValueError('invalid public key length')
    return raw


def config():
    return {'actor':'frozen-rules-v1','executor':'process-capability-v2',
            'behavior_suite':'adversarial-v2','bootstrap_reps':12000,
            'alpha_total':0.05,'holdout_policy':'fresh-seeds-one-shot'}


def public_rules(plan):
    return {'cluster_count':len(plan['seeds']),**{k:plan[k] for k in ('min_improvement','max_mean_regression','max_correction_failures',
                                'min_cluster_seeds','cases_per_seed','config')}}


def validate_rules(rules):
    fields={'min_improvement','max_mean_regression','max_correction_failures',
            'min_cluster_seeds','cases_per_seed','config','cluster_count'}
    if type(rules)is not dict or set(rules)!=fields:raise ValueError('evaluation rule schema')
    if type(rules['min_improvement'])is not float or not math.isfinite(rules['min_improvement']) or rules['min_improvement']<0:
        raise ValueError('invalid improvement threshold')
    if type(rules['max_mean_regression'])is not float or rules['max_mean_regression']!=0.0:
        raise ValueError('mean regression must be zero')
    if type(rules['max_correction_failures'])is not int or rules['max_correction_failures']!=0:
        raise ValueError('correction regression must be zero')
    if type(rules['min_cluster_seeds'])is not int or not 5<=rules['min_cluster_seeds']<=64:
        raise ValueError('invalid seed count')
    if type(rules['cluster_count'])is not int or not rules['min_cluster_seeds']<=rules['cluster_count']<=64:
        raise ValueError('invalid committed cluster count')
    if type(rules['cases_per_seed'])is not int or not 30<=rules['cases_per_seed']<=2000:
        raise ValueError('invalid episode count')
    if rules['config']!=config() or any(type(rules['config'][k])is not type(v) for k,v in config().items()):
        raise ValueError('invalid evaluation configuration')
