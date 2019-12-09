import os
from pathlib import Path


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = ROOT_DIR / "data"