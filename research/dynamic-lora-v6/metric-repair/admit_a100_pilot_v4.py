"""Create a fresh self-signed development admission; does not provision Colab."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))

from saal_rsi.adaptive_contract_v4 import adapter_spec,new_job,worker_identity
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_evaluator_v4 import validate_protocol,validate_sources
from saal_rsi.adaptive_registry_v4 import AdaptiveRegistryV4
from saal_rsi.adaptive_runtime_v4 import runtime_requirements
from saal_rsi.model_contract import sign,verify
from saal_rsi.numerical_metrics_v2 import POLICY,POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec,registry_path
from saal_rsi.spec import canonical,sha

CAMPAIGN_ID='saal_v097_a100_diagnostic_v4_joborderfix_20261017'
SEED=47
METHODS=('fixed-r8','fixed-r12','adalora-static-r12','adalora-12-to-8')
MAX_COMPUTE_UNITS=80
TOTAL_LIFECYCLE_SECONDS=14000
MAX_RUN_SECONDS=1800
RESERVED_SECONDS=3200


def colab_cli_version():
    result=subprocess.run(['colab','version'],check=True,capture_output=True,text=True,timeout=15)
    match=re.fullmatch(r'Version: ([0-9]+\.[0-9]+\.[0-9]+)\s*',result.stdout)
    if not match:raise ValueError('unrecognized Colab CLI version response')
    return match.group(1)


def source_inventory():
    files=set()
    from saal_rsi.adaptive_contract_v4 import SOURCES
    files.update('src/saal_rsi/'+name for name in SOURCES)
    files.update({
        'src/saal_rsi/adaptive_campaign_v4.py',
        'src/saal_rsi/adaptive_colab_v4.py',
        'src/saal_rsi/adaptive_campaign.py',
        'src/saal_rsi/adaptive_compute_budget_v4.py',
        'src/saal_rsi/adaptive_evaluator_v4.py',
        'src/saal_rsi/adaptive_registry_v4.py',
        'src/saal_rsi/model_monitor.py',
        'src/saal_rsi/model_monitor_v3.py',
        'configs/dependency-locks/pretrained-linux-x86_64-py313.txt',
        'configs/pretrained/registry.json',
        'configs/pretrained/tensor_inventories.json',
        'research/evidence-closure-v5/dataset.json',
        'research/dynamic-lora-v6/metric-repair/admit_a100_pilot_v4.py',
        'research/dynamic-lora-v6/metric-repair/report_a100_pilot_v4.py',
    })
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)}


def build_payloads():
    population=json.loads((ROOT/'research/evidence-closure-v5/dataset.json').read_text())
    dataset=training_projection(population)
    runtime=runtime_requirements('A100','HIGH_RAM')
    model=model_spec('answerdotai/ModernBERT-base',method='none',rank=0,
        backend='colab',precision='float32',budget=CAMPAIGN_ID,max_tokens=256)
    budget_base={'campaign_id':CAMPAIGN_ID,'max_runs':len(METHODS),
        'total_lifecycle_seconds':TOTAL_LIFECYCLE_SECONDS,
        'automatic_training_retries':0,'financial_cap':False}
    jobs=[];descriptors=[]
    for method in METHODS:
        name=f'{method.replace("-","_")}_{SEED}'
        budget={**budget_base,'run_id':name,'reserved_seconds':RESERVED_SECONDS}
        adapter=adapter_spec(method,total_steps=1000,tinit=100,tfinal=200,delta=10)
        job=new_job(model,dataset,adapter,seed=SEED,budget=budget,steps=1000,
            batch_size=8,accumulation=2,max_seconds=MAX_RUN_SECONDS,campaign_stage='PILOT',
            learning_rate=0.0003,head_learning_rate=0.0003,checkpoint_interval=100,
            runtime_requirements=runtime)
        jobs.append(job)
        descriptors.append({'name':name,'method':method,'seed':SEED,'job_digest':sha(job)})
    budget={'campaign_id':CAMPAIGN_ID,'max_runs':len(METHODS),
        'total_lifecycle_seconds':TOTAL_LIFECYCLE_SECONDS,
        'max_provisioning_attempts':1,'automatic_training_retries':0,
        'financial_cap':False,'max_compute_units':MAX_COMPUTE_UNITS}
    sources=source_inventory()
    protocol={'schema':'saal-adaptive-campaign-v4','stage':'PILOT','budget':budget,
        'job_digests':[sha(job) for job in jobs],'runs':descriptors,
        'worker_identities':{'v4':worker_identity()},
        'numerical_metric_policy':POLICY,'numerical_metric_policy_digest':POLICY_DIGEST,
        'colab_cli_version':colab_cli_version(),'source_files':sources,
        'required_source_files':sorted(sources),
        'independent_generalization':'NOT_ESTABLISHED',
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
        'runtime_requirements':runtime,
        'data_review_status':'UNREVIEWED_PUBLIC_DIAGNOSTIC',
        'scope':'ENGINEERING_ONLY_NO_BEHAVIORAL_CLAIM',
        'signer_status':'SELF_SIGNED_DEVELOPMENT_ONLY_NOT_INDEPENDENT'}
    validate_protocol(protocol)
    if sum(job['budget']['reserved_seconds'] for job in jobs)>TOTAL_LIFECYCLE_SECONDS:
        raise ValueError('per-run reservations exceed campaign lifecycle')
    validate_sources(protocol,ROOT)
    return jobs,protocol


def prepare(output):
    output=Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('preserve every previous admission; output must be new')
    jobs,protocol=build_payloads()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key=Ed25519PrivateKey.generate()  # In-memory development signer; never serialized or uploaded.
    pin=key.public_key().public_bytes_raw().hex()
    signed_jobs=[sign(job,key) for job in jobs]
    signed_protocol=sign(protocol,key)
    if verify(signed_protocol,pin)!=protocol or any(verify(e,pin)!=j for e,j in zip(signed_jobs,jobs)):
        raise ValueError('fresh development signature self-verification failed')
    output.mkdir(mode=0o700,parents=True)
    (output/'jobs').mkdir(mode=0o700)
    for run,envelope in zip(protocol['runs'],signed_jobs):
        (output/'jobs'/(run['name']+'-job.json')).write_bytes(canonical(envelope))
    (output/'signed_protocol.json').write_bytes(canonical(signed_protocol))
    (output/'controller.pub').write_text(pin+'\n')
    (output/'authorization-record.json').write_bytes(canonical({
        'schema':'saal-user-authorization-record-v2',
        'authorization':'Direct user authorization to run the bounded Colab A100/high-RAM engineering diagnostic.',
        'signer_status':protocol['signer_status'],
        'scope':protocol['scope'],'data_review_status':protocol['data_review_status'],
        'methods':list(METHODS),'seed':SEED,'runs':len(jobs),
        'max_compute_units':MAX_COMPUTE_UNITS,'prior_attempts_reused':False,
        'private_key_persisted':False,'private_key_uploaded':False,
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}))
    registry=AdaptiveRegistryV4(output/'budget.sqlite3',pin,signed_protocol,signed_jobs,create=True)
    try:
        registry.reserve_all(signed_jobs)
        (output/'preprovision-registry.json').write_bytes(canonical(registry.records()))
    finally:registry.close()
    (output/'admission-manifest.json').write_bytes(canonical({
        'schema':'saal-a100-engineering-admission-v4','campaign_digest':sha(protocol),
        'signed_protocol_digest':sha(signed_protocol),'job_digests':[sha(j) for j in jobs],
        'runtime_requirements':protocol['runtime_requirements'],
        'compute_units_cap':MAX_COMPUTE_UNITS,'signer_status':protocol['signer_status'],
        'state':'SIGNED_NOT_PROVISIONED','qualification_status':'NOT_EVALUATED',
        'external_deployment_allowed':False}))
    return output


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: admit_a100_pilot_v4.py NEW_ADMISSION_DIRECTORY')
    output=prepare(sys.argv[1])
    print(json.dumps({'admission':str(output),'run_count':len(METHODS),
        'runtime':'A100_HIGH_RAM','compute_units_cap':MAX_COMPUTE_UNITS,
        'signer_status':'SELF_SIGNED_DEVELOPMENT_ONLY_NOT_INDEPENDENT',
        'state':'SIGNED_NOT_PROVISIONED','qualification_status':'NOT_EVALUATED',
        'external_deployment_allowed':False}))
