#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Export PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p ./NEED_to_PREPARE/cm

bash ./tools/pdb_to_cm/pdb_to_cm.sh

mkdir -p ./NEED_to_PREPARE/pkl

python ./tools/feature_extract/feature_extra.py

python ./tools/Predict.py

rm -rf ./NEED_to_PREPARE/cm&&rm -rf ./NEED_to_PREPARE/pkl

