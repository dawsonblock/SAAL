import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-40ffb0dade90')
shutil.rmtree('/content/run-00-40ffb0dade90-transfer')
Path('/content/run-00-40ffb0dade90.zip').unlink()
