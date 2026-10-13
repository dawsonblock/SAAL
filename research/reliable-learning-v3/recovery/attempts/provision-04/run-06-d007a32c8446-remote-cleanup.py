import shutil
from pathlib import Path
shutil.rmtree('/content/run-06-d007a32c8446')
shutil.rmtree('/content/run-06-d007a32c8446-transfer')
Path('/content/run-06-d007a32c8446.zip').unlink()
