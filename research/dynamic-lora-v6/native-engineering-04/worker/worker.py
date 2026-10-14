from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from saal_rsi.adaptive_worker import main
main()
