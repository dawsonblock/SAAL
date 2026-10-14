import shutil
from pathlib import Path
shutil.rmtree('/content/run-05-1b6185b0ef56')
shutil.rmtree('/content/run-05-1b6185b0ef56-transfer')
Path('/content/run-05-1b6185b0ef56.zip').unlink()
