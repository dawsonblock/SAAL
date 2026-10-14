import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-875f98391b45')
shutil.rmtree('/content/run-00-875f98391b45-transfer')
Path('/content/run-00-875f98391b45.zip').unlink()
