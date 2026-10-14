import shutil
from pathlib import Path
shutil.rmtree('/content/run-03-20f86041551b')
shutil.rmtree('/content/run-03-20f86041551b-transfer')
Path('/content/run-03-20f86041551b.zip').unlink()
