import shutil
from pathlib import Path
shutil.rmtree('/content/run-04-09d66c1d8335')
shutil.rmtree('/content/run-04-09d66c1d8335-transfer')
Path('/content/run-04-09d66c1d8335.zip').unlink()
