import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-387472842fa8')
shutil.rmtree('/content/run-01-387472842fa8-transfer')
Path('/content/run-01-387472842fa8.zip').unlink()
