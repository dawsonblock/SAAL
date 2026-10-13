import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-c69099cf9688')
shutil.rmtree('/content/run-00-c69099cf9688-transfer')
Path('/content/run-00-c69099cf9688.zip').unlink()
