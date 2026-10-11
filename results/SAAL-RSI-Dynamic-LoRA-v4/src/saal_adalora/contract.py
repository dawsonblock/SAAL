"""Explicit experimental AdaLoRA identity; never reinterpret native v5 jobs."""
from pathlib import Path
import hashlib
from saal_rsi.spec import sha
from saal_rsi.pretrained_contract import validate_job,worker_identity
FIXED_PIN='0352113286a06e211a809862970ea405c6d140578755850875e591881ceb7bf8'
FIXED_DIGEST='sha256:c8408a3db01f0760132c32ba0fab2d8ddea921d3ecedc0cc51f6f567df503e80'
SCREEN={'complete_pair_accuracy':.9,'worst_family_accuracy':.8,'required_class_coverage':4,'unauthorized_proposals':0,'invalid_outputs':0}
ADAPTER={'kind':'peft-adalora-masked-v1','init_r':8,'target_r':4,'tinit':100,'tfinal':100,'deltaT':10,'beta1':.85,'beta2':.85,'orth_reg_weight':.5,'lora_alpha':16,'lora_dropout':.05,'total_step':600,'serialization':'full-initial-rank-matrices-plus-mask-metadata'}
def source_identity():return sha({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
def validate(job):
    if type(job)is not dict or set(job)!={'schema','native_configuration_reference','dynamic_adapter','dynamic_source_identity','base_worker_identity','fixed_comparison_digest','fixed_comparison_envelope','fixed_comparison_pin','screen','resource_budget_id','reserved_seconds','qualification_status','external_deployment_allowed'}:raise ValueError('dynamic contract fields')
    if job['schema']!='saal-adalora-research-job-v1' or job['dynamic_adapter']!=ADAPTER or job['screen']!=SCREEN or job['reserved_seconds']!=2300 or job['resource_budget_id']!='dynamic4_'+source_identity()[7:23] or job['dynamic_source_identity']!=source_identity() or job['base_worker_identity']!=worker_identity() or job['qualification_status']!='NOT_EVALUATED' or job['external_deployment_allowed']is not False:raise ValueError('dynamic authority/config identity')
    native=job['native_configuration_reference'];validate_job(native)
    if native['model_spec']['model_repository']!='answerdotai/ModernBERT-base' or native['model_spec']['adapter_type']!='lora' or native['model_spec']['adapter_rank']!=4 or native['model_spec']['compute_backend']!='local' or native['model_spec']['precision']!='float32' or native['seed']!=47 or native['steps']!=600 or native['batch_size']!=8 or native['learning_rate']!=.0003 or native['max_seconds']!=1800:raise ValueError('bounded matched training reference')
    from saal_rsi.model_contract import verify
    if job['fixed_comparison_pin']!=FIXED_PIN or verify(job['fixed_comparison_envelope'],FIXED_PIN)!=native or job['fixed_comparison_digest']!=sha(native) or sha(native)!=FIXED_DIGEST:raise ValueError('fixed comparison substitution')
    # The native reference is a configuration/comparison identity, not an executed v5 AdaLoRA job.
    return native
