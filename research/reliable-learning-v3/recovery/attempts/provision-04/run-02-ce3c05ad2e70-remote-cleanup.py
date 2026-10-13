import shutil
from pathlib import Path
shutil.rmtree('/content/run-02-ce3c05ad2e70')
shutil.rmtree('/content/run-02-ce3c05ad2e70-transfer')
Path('/content/run-02-ce3c05ad2e70.zip').unlink()
