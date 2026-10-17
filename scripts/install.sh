#!/usr/bin/env bash
set -euo pipefail

RATIO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RATIO_ROOT"
RATIO_PYTHON="${RATIO_PYTHON:-python}"

"$RATIO_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Activate a Python 3.12 environment"'
command -v git >/dev/null
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
command -v nvcc >/dev/null || {
    printf '%s\n' 'CUDA compiler not found. Install a compatible CUDA toolkit and set CUDA_HOME or add nvcc to PATH.' >&2
    exit 1
}
export MAX_JOBS="${MAX_JOBS:-4}"
unset VLLM_USE_PRECOMPILED

"$RATIO_PYTHON" tools/prepare_dependencies.py --only vllm lighteval fast-hadamard-transform
"$RATIO_PYTHON" -m pip install 'setuptools>=74.1.1' 'setuptools-scm>=8' wheel packaging 'cmake>=3.26,<4'
"$RATIO_PYTHON" -m pip install -c requirements.txt -e ./third-party/fast-hadamard-transform --no-build-isolation
"$RATIO_PYTHON" -m pip install -c requirements.txt -e ./third-party/vllm --no-build-isolation
"$RATIO_PYTHON" -m pip install -c requirements.txt -e './third-party/lighteval[math]'
"$RATIO_PYTHON" -m pip install -c requirements.txt \
    'protobuf==3.20.3' 'google-api-core==2.29.0' \
    'googleapis-common-protos==1.73.0' 'proto-plus==1.27.1' \
    'opentelemetry-proto==1.27.0'
