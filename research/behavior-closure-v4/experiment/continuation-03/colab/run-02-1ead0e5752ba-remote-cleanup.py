import shutil
from pathlib import Path
shutil.rmtree('/content/run-02-1ead0e5752ba')
shutil.rmtree('/content/run-02-1ead0e5752ba-transfer')
Path('/content/run-02-1ead0e5752ba.zip').unlink()
