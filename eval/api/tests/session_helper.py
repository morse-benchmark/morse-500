"""Build a complete setup/session_env.sh environment for script tests."""
import os
from pathlib import Path

API = Path(__file__).resolve().parents[1]


def session_env(root, **overrides):
    root = Path(root)
    dirs = {
        "RESULTS_ROOT": root / "eval/results", "LOG_DIR": root / "logs",
        "FRAME_CACHE_ROOT": root / "frame_cache", "AGENT_BIN": root / "agent-bin",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    data_csv = root / "session-test.csv"
    data_csv.touch()
    env = dict(
        os.environ, MORSE_SESSION_ENV="1", API_DIR=str(API), REPO=str(root),
        DATA_CSV=str(data_csv), VIDEO_ROOT=str(root), VENV=str(root / "venv"),
        PYTHON=str(root / "venv/bin/python"),
        CODEX_BINARY=str(dirs["AGENT_BIN"] / "codex"),
        CLAUDE_BINARY=str(dirs["AGENT_BIN"] / "claude"),
        CODEX_CODE_MODE_HOST=str(dirs["AGENT_BIN"] / "codex-code-mode-host"),
        **{name: str(path) for name, path in dirs.items()},
    )
    env.update({name: str(value) for name, value in overrides.items()})
    return env
