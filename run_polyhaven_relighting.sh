#!/bin/bash
set -e

DATA_ROOT="/data/polyhaven_lvsm/test"
METADATA_DIR="relight_metadata"
DOWNSAMPLE=1
NUM_PROCESSES=8

for meta_file in "$METADATA_DIR"/*.json; do
    scene=$(basename "$meta_file" .json)
    echo "=== Processing: $scene ==="
    accelerate launch --num_processes "$NUM_PROCESSES" produce_gs_relightings.py \
        --dataset_type polyhaven \
        --data_root "$DATA_ROOT" \
        --relight_metadata "$meta_file" \
        --downsample "$DOWNSAMPLE"
    echo "=== Done: $scene ==="
done
