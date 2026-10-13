import shutil
from pathlib import Path
shutil.rmtree('/content/run-04-84e3801a3158')
shutil.rmtree('/content/run-04-84e3801a3158-transfer')
Path('/content/run-04-84e3801a3158.zip').unlink()
