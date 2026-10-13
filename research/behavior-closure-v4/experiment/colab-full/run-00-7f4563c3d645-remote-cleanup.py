import shutil
from pathlib import Path
shutil.rmtree('/content/run-00-7f4563c3d645')
shutil.rmtree('/content/run-00-7f4563c3d645-transfer')
Path('/content/run-00-7f4563c3d645.zip').unlink()
