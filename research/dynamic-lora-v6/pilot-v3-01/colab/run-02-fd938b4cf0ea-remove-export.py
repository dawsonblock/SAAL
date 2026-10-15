import shutil
from pathlib import Path
shutil.rmtree('/content/run-02-fd938b4cf0ea')
shutil.rmtree('/content/run-02-fd938b4cf0ea-transfer')
Path('/content/run-02-fd938b4cf0ea.zip').unlink()
