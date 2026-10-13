import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-d8edcfdf8072')
shutil.rmtree('/content/run-00-d8edcfdf8072-transfer')
Path('/content/run-00-d8edcfdf8072.zip').unlink()
