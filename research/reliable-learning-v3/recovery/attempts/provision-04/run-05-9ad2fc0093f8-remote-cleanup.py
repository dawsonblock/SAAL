import shutil
from pathlib import Path
shutil.rmtree('/content/run-05-9ad2fc0093f8')
shutil.rmtree('/content/run-05-9ad2fc0093f8-transfer')
Path('/content/run-05-9ad2fc0093f8.zip').unlink()
