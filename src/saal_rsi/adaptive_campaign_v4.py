"""Allowlisted v4 worker staging; staging does not admit or execute a campaign."""
from pathlib import Path

from .pretrained_contract import registry_path
from .adaptive_artifacts import read_json
from .model_monitor import stream_command


def stage_worker(destination):
    """Stage only the exact v4 worker inventory, without keys or unrelated modules."""
    from .adaptive_contract_v4 import SOURCES
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('worker destination must be new')
    package = destination/'src/saal_rsi'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    base = Path(__file__).with_name('adaptive_contract_v4.py').parent
    for name in SOURCES:
        path = base/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('allowlisted worker source missing or unsafe: '+name)
        (package/name).write_bytes(path.read_bytes())
    configs = destination/'configs/pretrained'
    configs.mkdir(parents=True)
    for name in ('registry.json', 'tensor_inventories.json'):
        source = registry_path().with_name(name)
        (configs/name).write_bytes(source.read_bytes())
    locks = destination/'configs/dependency-locks'
    locks.mkdir(parents=True)
    for name in ('pretrained-linux-x86_64-py313.txt', 'pretrained-macos-arm64-py312.txt'):
        source = registry_path().parents[1]/'dependency-locks'/name
        (locks/name).write_bytes(source.read_bytes())
    bootstrap = """from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
import argparse
from saal_rsi.pretrained_contract import job_bytes
from saal_rsi.spec import parse_unique_json
parser=argparse.ArgumentParser(add_help=False); parser.add_argument('--job'); known,_=parser.parse_known_args()
schema=parse_unique_json(job_bytes(known.job))['payload']['schema'] if known.job else None
if schema != 'saal-adaptive-job-v4': raise ValueError('v4 worker accepts only v4 signed jobs')
from saal_rsi.adaptive_worker_v4 import main
main()
"""
    (destination/'worker.py').write_text(bootstrap)


def replay_command(config, bundle, job_file, destination):
    """Replay a frozen v4 bundle in a separate local process; never trains."""
    import sys

    bundle, job_file, destination = Path(bundle), Path(job_file), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('preserve prior v4 replay receipt')
    script = Path(config['workspace'])/(destination.stem+'-replay-v4.py')
    root = str(Path(__file__).resolve().parents[1])
    script.write_text(
        'import sys\nfrom pathlib import Path\n'+f'sys.path.insert(0,{root!r})\n'+
        'from saal_rsi.adaptive_replay_v4 import fresh_local_replay\n'+
        'from saal_rsi.model_contract import verify\n'+
        'from saal_rsi.spec import canonical,parse_unique_json\n'+
        f"pin={config['pin']!r}\n"+
        f"job=verify(parse_unique_json(Path({str(job_file)!r}).read_bytes()),pin)\n"+
        f"value=fresh_local_replay({str(bundle)!r},job,pin,{config['cache']!r})\n"+
        f"with Path({str(destination)!r}).open('xb') as output:output.write(canonical(value))\n"+
        'from saal_rsi.adaptive_artifacts_v4 import validate_bundle\n'+
        f"validate_bundle({str(bundle)!r},job,pin,require_replay=True)\n"+
        'if not value["verified"]:raise ValueError("v4 local checkpoint replay failed")\n')
    stream_command([sys.executable, '-I', str(script)], destination.with_suffix('.log'), 600)
