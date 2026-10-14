import shutil
from pathlib import Path
shutil.rmtree('/content/run-08-641576adfbcd')
shutil.rmtree('/content/run-08-641576adfbcd-transfer')
Path('/content/run-08-641576adfbcd.zip').unlink()
