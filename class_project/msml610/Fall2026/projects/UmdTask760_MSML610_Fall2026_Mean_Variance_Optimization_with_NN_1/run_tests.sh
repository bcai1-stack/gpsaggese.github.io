#!/bin/bash
# Validate in a fresh container, without affecting a running Jupyter session.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$PROJECT_DIR" rev-parse --show-toplevel)
RELATIVE_DIR=${PROJECT_DIR#"$GIT_ROOT"/}
source "$PROJECT_DIR/docker_name.sh"

FULL_VALIDATION=0
if [[ $# -eq 1 && "$1" == "--notebooks" ]]; then
    FULL_VALIDATION=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--notebooks]" >&2
    exit 2
fi

docker run --rm \
    -v "$GIT_ROOT:/git_root" \
    -w "/git_root/$RELATIVE_DIR" \
    -e MPLCONFIGDIR=/tmp/mvo-mpl \
    -e MVO_FULL_VALIDATION="$FULL_VALIDATION" \
    "$FULL_IMAGE_NAME" \
    jupyter nbconvert --execute --to notebook --inplace \
        --ExecutePreprocessor.timeout=1800 mvo.validation.ipynb
