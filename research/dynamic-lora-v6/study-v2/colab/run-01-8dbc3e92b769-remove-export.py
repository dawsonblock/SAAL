import shutil
from pathlib import Path
shutil.rmtree('/content/run-01-8dbc3e92b769')
shutil.rmtree('/content/run-01-8dbc3e92b769-transfer')
Path('/content/run-01-8dbc3e92b769.zip').unlink()
