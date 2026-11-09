"""Compatibility entry point for generalized direct-model and agent estimates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimates.estimate_agent_costs import *
from estimates.estimate_agent_costs import main


if __name__ == "__main__":
    main()
