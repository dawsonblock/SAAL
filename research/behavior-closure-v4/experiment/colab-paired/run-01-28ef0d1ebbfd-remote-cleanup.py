import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-28ef0d1ebbfd')
shutil.rmtree('/content/run-01-28ef0d1ebbfd-transfer')
Path('/content/run-01-28ef0d1ebbfd.zip').unlink()
