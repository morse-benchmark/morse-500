#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash eval/qwen_eval/setup/validate.sh --stage setup|gpu|server"
}
STAGE=""
while (( $# )); do
    case "$1" in
        --stage)
            if (( $# < 2 )); then
                usage >&2
                exit 2
            fi
            STAGE="$2"
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$STAGE" in
    setup|gpu|server) ;;
    *) usage >&2; exit 2 ;;
esac
QWEN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "$QWEN_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.qwen_venv/bin/python}"
[[ -x "$PYTHON" ]] || {
    echo "Python interpreter missing: $PYTHON; run eval/qwen_eval/setup/setup_env.sh." >&2
    exit 3
}
export HF_HOME="${HF_HOME:-/fs/nexus-scratch/dgjersta/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "$STAGE" == gpu ]]; then
    export CUDA_HOME="${CUDA_HOME:-/opt/common/cuda/cuda-13.0.2}"
    export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || {
        echo "CUDA compiler not found at $CUDA_HOME/bin/nvcc; set CUDA_HOME to a complete compatible CUDA toolkit." >&2
        exit 3
    }
    command -v ninja >/dev/null || {
        echo "Ninja is missing from PATH; use the installed .qwen_venv environment." >&2
        exit 3
    }
    "$CUDA_HOME/bin/nvcc" --version
    ninja --version
fi
"$PYTHON" "$QWEN_DIR/setup/validate.py" --stage "$STAGE"
