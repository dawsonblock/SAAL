import json,subprocess,zipfile
from pathlib import Path
with zipfile.ZipFile('/content/saal_worker.zip') as z:
    if set(z.namelist())!=set(['worker.py', 'saal_rsi/__init__.py', 'saal_rsi/spec.py', 'saal_rsi/validation.py', 'saal_rsi/model_contract.py', 'saal_rsi/dataset_registry.py', 'saal_rsi/neural_contract.py', 'saal_rsi/model_architectures.py', 'saal_rsi/neural_training.py', 'saal_rsi/neural_worker.py']):raise ValueError('worker source inventory')
    z.extractall('/content/saal-neural')
command=['/content/saal-env/bin/python','-I','/content/saal-neural/worker.py','--job','/content/saal_job.json','--public-key','2fb28f528dc3e2e24d04600c32b5dfb89e17d3c02f104c0f6649c3dd4c0903c6','--out','/content/saal_result_aaa85f7589fe','--require-gpu','T4']
command+=['--baseline','/content/saal_baseline.safetensors']
subprocess.run(command,check=True,timeout=630)
with zipfile.ZipFile('/content/saal_result_aaa85f7589fe.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['adapter.safetensors', 'artifact_inventory.json', 'baseline.safetensors', 'candidate.safetensors', 'checkpoint_replay.json', 'checkpoint_selection.json', 'cleanup_evidence.json', 'dataset_manifest.json', 'experiment_manifest.json', 'model_spec.json', 'resource_accounting.json', 'result.json', 'runtime_metrics.json', 'signed_training_job.json', 'tensor_inventory.json', 'training_metrics.jsonl']:z.write(Path('/content/saal_result_aaa85f7589fe')/name,name)
