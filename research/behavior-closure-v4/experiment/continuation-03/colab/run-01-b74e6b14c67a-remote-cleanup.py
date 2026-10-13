import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-b74e6b14c67a')
shutil.rmtree('/content/run-01-b74e6b14c67a-transfer')
Path('/content/run-01-b74e6b14c67a.zip').unlink()
