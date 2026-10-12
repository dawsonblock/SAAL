"""Reuse the verified v3 QA payload builder in a new evidence namespace."""
from pathlib import Path
import importlib.util
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
source=ROOT/'research/reliable-learning-v3/tools/colab_tests.py'
spec=importlib.util.spec_from_file_location('saal_verified_v3_qa',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=Path(__file__).parent
module.REPO=ROOT

if __name__=='__main__':module.run('colab-tests')
