import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-53dd56d784a8')
shutil.rmtree('/content/run-00-53dd56d784a8-transfer')
Path('/content/run-00-53dd56d784a8.zip').unlink()
