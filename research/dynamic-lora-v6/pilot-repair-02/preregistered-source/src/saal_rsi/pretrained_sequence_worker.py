"""Separately signed and reviewed T4 sequence execution; public inputs only."""
from pathlib import Path
import argparse,json,os
from .spec import parse_unique_json,canonical
from .model_contract import verify
from .pretrained_contract import validate_job
from .pretrained_continual import run_sequence,validate_sequence,sequence_worker_identity
from .pretrained_registry import fetch_snapshot
from .learning_backend import verify_dependencies

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--protocol',required=True);p.add_argument('--pin',required=True);p.add_argument('--learning-report',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TOKENIZERS_PARALLELISM']='false';verify_dependencies()
    job_envelope=parse_unique_json(Path(a.job).read_bytes());job=verify(job_envelope,a.pin);validate_job(job)
    protocol_envelope=parse_unique_json(Path(a.protocol).read_bytes());protocol=verify(protocol_envelope,a.pin);validate_sequence(protocol,job)
    import torch
    torch.set_num_threads(2)
    if job['model_spec']['compute_backend']!='colab' or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('sequence requires admitted T4')
    if torch.cuda.mem_get_info()[0]<6*1024**3:raise ValueError('sequence GPU free-memory admission')
    s=job['model_spec'];snapshot=fetch_snapshot(a.cache,s['model_repository'],s['model_revision'])
    result=run_sequence(job_envelope,protocol_envelope,a.pin,snapshot,a.out,parse_unique_json(Path(a.learning_report).read_bytes()))
    result.update(schema='saal-sequence-result-v1',worker_identity=sequence_worker_identity(),gpu=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    root=Path(a.out)
    for name,value in [('result.json',result),('signed_job.json',job_envelope),('signed_protocol.json',protocol_envelope)]: (root/name).write_bytes(canonical(value))
    from .pretrained_registry import file_hash
    inventory={p.name:{'size':p.stat().st_size,'sha256':file_hash(p)} for p in root.iterdir() if p.is_file()}
    (root/'inventory.json').write_bytes(canonical({'schema':'saal-sequence-inventory-v1','files':inventory}))
    print(json.dumps({'monitor':'sequence_complete','method':protocol['method'],'replay_ratio':protocol['replay_ratio'],'matrix':result['accuracy_matrix'],'qualification_status':'NOT_EVALUATED'}),flush=True)

if __name__=='__main__':main()
