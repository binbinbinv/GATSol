#!/bin/sh

mkdir -p ./NEED_to_PREPARE/cm

bash ./tools/pdb_to_cm/pdb_to_cm.sh

mkdir -p ./NEED_to_PREPARE/pkl

python ./tools/feature_extract/feature_extra.py

python ./tools/Predict.py

rm -rf ./NEED_to_PREPARE/cm&&rm -rf ./NEED_to_PREPARE/pkl