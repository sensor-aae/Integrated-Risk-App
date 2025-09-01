import sys
from pathlib import Path

# Add the repo root to sys.path so "import risk_engine" works in CI
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
