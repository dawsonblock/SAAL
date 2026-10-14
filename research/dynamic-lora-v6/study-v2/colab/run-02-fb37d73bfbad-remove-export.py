import shutil
from pathlib import Path
shutil.rmtree('/content/run-02-fb37d73bfbad')
shutil.rmtree('/content/run-02-fb37d73bfbad-transfer')
Path('/content/run-02-fb37d73bfbad.zip').unlink()
