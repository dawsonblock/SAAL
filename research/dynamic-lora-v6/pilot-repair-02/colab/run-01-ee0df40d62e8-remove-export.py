import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-ee0df40d62e8')
shutil.rmtree('/content/run-01-ee0df40d62e8-transfer')
Path('/content/run-01-ee0df40d62e8.zip').unlink()
