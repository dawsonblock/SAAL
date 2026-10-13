import shutil
from pathlib import Path
shutil.rmtree('/content/run-02-c553fcf68a09')
shutil.rmtree('/content/run-02-c553fcf68a09-transfer')
Path('/content/run-02-c553fcf68a09.zip').unlink()
