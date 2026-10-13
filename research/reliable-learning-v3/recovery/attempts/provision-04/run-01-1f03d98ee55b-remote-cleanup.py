import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-1f03d98ee55b')
shutil.rmtree('/content/run-01-1f03d98ee55b-transfer')
Path('/content/run-01-1f03d98ee55b.zip').unlink()
