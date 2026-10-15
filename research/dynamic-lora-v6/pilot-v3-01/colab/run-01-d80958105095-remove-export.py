import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-d80958105095')
shutil.rmtree('/content/run-01-d80958105095-transfer')
Path('/content/run-01-d80958105095.zip').unlink()
