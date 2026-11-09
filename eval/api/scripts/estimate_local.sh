#!/usr/bin/env bash
set -euo pipefail

export ACTION=estimate
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/lib/run_eval.sh"
