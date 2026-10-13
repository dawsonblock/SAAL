import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-610c87b9e1e4')
shutil.rmtree('/content/run-00-610c87b9e1e4-transfer')
Path('/content/run-00-610c87b9e1e4.zip').unlink()
