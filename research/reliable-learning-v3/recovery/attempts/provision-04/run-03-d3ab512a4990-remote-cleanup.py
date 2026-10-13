import shutil
from pathlib import Path
shutil.rmtree('/content/run-03-d3ab512a4990')
shutil.rmtree('/content/run-03-d3ab512a4990-transfer')
Path('/content/run-03-d3ab512a4990.zip').unlink()
