import shutil
from pathlib import Path
shutil.rmtree('/content/run-06-2f07f33788ed')
shutil.rmtree('/content/run-06-2f07f33788ed-transfer')
Path('/content/run-06-2f07f33788ed.zip').unlink()
