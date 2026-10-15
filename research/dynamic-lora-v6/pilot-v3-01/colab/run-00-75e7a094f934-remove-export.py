import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-75e7a094f934')
shutil.rmtree('/content/run-00-75e7a094f934-transfer')
Path('/content/run-00-75e7a094f934.zip').unlink()
