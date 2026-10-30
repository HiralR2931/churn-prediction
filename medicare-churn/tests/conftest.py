import sys
from pathlib import Path

# Run from anywhere: make the project's ``src`` package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
