import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-1b97c12fa84a')
shutil.rmtree('/content/run-00-1b97c12fa84a-transfer')
Path('/content/run-00-1b97c12fa84a.zip').unlink()
