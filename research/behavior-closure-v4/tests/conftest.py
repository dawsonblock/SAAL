"""Standalone project tests resolve the source tree without editable installation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
