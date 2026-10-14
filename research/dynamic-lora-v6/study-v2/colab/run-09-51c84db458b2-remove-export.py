import shutil
from pathlib import Path
shutil.rmtree('/content/run-09-51c84db458b2')
shutil.rmtree('/content/run-09-51c84db458b2-transfer')
Path('/content/run-09-51c84db458b2.zip').unlink()
