import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-896a2acf95a7')
shutil.rmtree('/content/run-00-896a2acf95a7-transfer')
Path('/content/run-00-896a2acf95a7.zip').unlink()
