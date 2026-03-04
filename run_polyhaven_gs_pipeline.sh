#!/bin/bash
set -e

# ============================================================
# Full pipeline: Relighting + 3DGS for polyhaven dataset
#
# Steps per scene:
#   1. Convert polyhaven metadata → COLMAP format (for initial GS)
#   2. Relight images with LightSwitch
#   3. Train initial GS on original scene
#   4. Fine-tune GS with relit images (appearance only)
#   5. Render relit GS
# ============================================================

DATA_ROOT="/data/polyhaven_lvsm/test"
METADATA_DIR="relight_metadata"
DOWNSAMPLE=1
NUM_PROCESSES=1
GS_RESOLUTION=1         # --resolution for gaussian-splatting train.py
GS_ITERATIONS=30000
GS_FINETUNE_ITERATIONS=40000
GUIDANCE="3"
SM_GUIDANCE="3"

export CUDA_VISIBLE_DEVICES=0

for meta_file in "$METADATA_DIR"/*.json; do
    scene_json=$(basename "$meta_file" .json)

    scene_name=$(python3 -c "import json; print(json.load(open('$meta_file'))['scene_name'])")
    relit_scene_name=$(python3 -c "import json; print(json.load(open('$meta_file'))['relit_scene_name'])")

    echo ""
    echo "============================================================"
    echo " Scene: $scene_name → Relit: $relit_scene_name"
    echo "============================================================"

    COLMAP_DIR="data/polyhaven_colmap/${scene_name}"
    GS_MODEL="gs_outputs/${scene_name}"
    RELIT_DIR="relighting_outputs/rm_${GUIDANCE}_${SM_GUIDANCE}/${scene_name}/${relit_scene_name}"
    RELIT_GS_MODEL="gs_outputs/relit_gs/${scene_name}/${relit_scene_name}"

    # ----------------------------------------------------------
    # Step 1: Convert polyhaven metadata → COLMAP format
    # ----------------------------------------------------------
    if [ ! -f "${COLMAP_DIR}/sparse/0/cameras.bin" ]; then
        echo "[Step 1] Converting polyhaven → COLMAP: $scene_name"
        python scripts/polyhaven_to_colmap.py \
            --data_root "$DATA_ROOT" \
            --scene_name "$scene_name" \
            --output_dir "$COLMAP_DIR" \
            --downsample "$DOWNSAMPLE"
    else
        echo "[Step 1] COLMAP data already exists: $COLMAP_DIR (skipping)"
    fi

    # ----------------------------------------------------------
    # Step 2: Relight images
    # ----------------------------------------------------------
    if [ ! -d "${RELIT_DIR}/images" ]; then
        echo "[Step 2] Relighting: $scene_name → $relit_scene_name"
        accelerate launch --num_processes "$NUM_PROCESSES" produce_gs_relightings.py \
            --dataset_type polyhaven \
            --data_root "$DATA_ROOT" \
            --relight_metadata "$meta_file" \
            --downsample "$DOWNSAMPLE"
    else
        echo "[Step 2] Relit images already exist: $RELIT_DIR (skipping)"
    fi

    # ----------------------------------------------------------
    # Step 3: Train initial GS on original scene
    # ----------------------------------------------------------
    if [ ! -f "${GS_MODEL}/chkpnt${GS_ITERATIONS}.pth" ]; then
        echo "[Step 3] Training initial GS: $scene_name"
        python gaussian-splatting/train.py \
            -s "$COLMAP_DIR" \
            -m "$GS_MODEL" \
            --images images \
            --resolution "$GS_RESOLUTION" \
            --checkpoint_iterations "$GS_ITERATIONS"
    else
        echo "[Step 3] GS checkpoint already exists: ${GS_MODEL}/chkpnt${GS_ITERATIONS}.pth (skipping)"
    fi

    # Verify initial GS (optional render)
    # python gaussian-splatting/render.py -m "$GS_MODEL"

    # ----------------------------------------------------------
    # Step 4: Fine-tune GS with relit images (freeze geometry)
    # ----------------------------------------------------------
    if [ ! -d "${RELIT_GS_MODEL}/point_cloud" ]; then
        echo "[Step 4] Fine-tuning GS with relit images"
        python gaussian-splatting/train.py \
            -s "$RELIT_DIR" \
            --start_checkpoint "${GS_MODEL}/chkpnt${GS_ITERATIONS}.pth" \
            --iterations "$GS_FINETUNE_ITERATIONS" \
            -m "$RELIT_GS_MODEL" \
            --images images \
            --resolution "$GS_RESOLUTION" \
            --position_lr_init 0.0 \
            --position_lr_final 0.0 \
            --opacity_lr 0.0 \
            --scaling_lr 0.0 \
            --rotation_lr 0.0
    else
        echo "[Step 4] Relit GS already exists: $RELIT_GS_MODEL (skipping)"
    fi

    # ----------------------------------------------------------
    # Step 5: Render relit GS
    # ----------------------------------------------------------
    echo "[Step 5] Rendering relit GS"
    python gaussian-splatting/render.py -m "$RELIT_GS_MODEL"

    echo "=== Done: $scene_name → $relit_scene_name ==="
done

echo ""
echo "All scenes complete."
