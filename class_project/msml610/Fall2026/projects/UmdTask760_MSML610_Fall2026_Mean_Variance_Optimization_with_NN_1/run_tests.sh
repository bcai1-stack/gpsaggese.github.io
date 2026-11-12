#!/bin/bash
# Validate in a fresh container, without affecting a running Jupyter session.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$PROJECT_DIR" rev-parse --show-toplevel)
RELATIVE_DIR=${PROJECT_DIR#"$GIT_ROOT"/}
source "$PROJECT_DIR/docker_name.sh"

docker run --rm \
    -v "$GIT_ROOT:/git_root" \
    -w "/git_root/$RELATIVE_DIR" \
    -e MPLCONFIGDIR=/tmp/mvo-mpl \
    "$FULL_IMAGE_NAME" bash -c '
        set -e
        python -m pip install -q scikit-learn==1.8.0 torch==2.8.0 cvxpy==1.9.2 cvxpylayers==1.2.0
        python validate.py "$@"
    ' bash "$@"
