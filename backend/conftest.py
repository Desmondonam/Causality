import sys
from pathlib import Path

# Let `from app... import ...` resolve when pytest is run from the repo
# root (`pytest backend/`) as well as from inside `backend/`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
