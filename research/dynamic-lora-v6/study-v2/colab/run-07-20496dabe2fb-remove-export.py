import shutil
from pathlib import Path
shutil.rmtree('/content/run-07-20496dabe2fb')
shutil.rmtree('/content/run-07-20496dabe2fb-transfer')
Path('/content/run-07-20496dabe2fb.zip').unlink()
