import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-d4da716a6b89')
shutil.rmtree('/content/run-00-d4da716a6b89-transfer')
Path('/content/run-00-d4da716a6b89.zip').unlink()
