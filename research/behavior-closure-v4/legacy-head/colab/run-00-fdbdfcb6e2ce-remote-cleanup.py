import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-fdbdfcb6e2ce')
shutil.rmtree('/content/run-00-fdbdfcb6e2ce-transfer')
Path('/content/run-00-fdbdfcb6e2ce.zip').unlink()
