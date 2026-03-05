#!/bin/bash
set -e

SCENES=(
  ceramic_vase_02_white_env_0
  marble_bust_01_env_2
  pot_enamel_01_white_env_0
  potted_plant_02_white_env_0
)

DATA_ROOT="/data/polyhaven_lvsm/test"
METADATA_DIR="data_samples/relight_metadata"
export CUDA_VISIBLE_DEVICES=0

for OBJ in "${SCENES[@]}"; do
  echo ""
  echo "============================================================"
  echo " Scene: $OBJ"
  echo "============================================================"

  if [ ! -f "$METADATA_DIR/$OBJ.json" ]; then
    echo "Skipping $OBJ: metadata not found"
    continue
  fi

  RELIT=$(python3 -c "import json; print(json.load(open('$METADATA_DIR/$OBJ.json'))['relit_scene_name'])")

  # Step 1: polyhaven → COLMAP
  if [ ! -f "data/polyhaven_colmap/$OBJ/sparse/0/cameras.bin" ]; then
    python scripts/polyhaven_to_colmap.py \
      --data_root "$DATA_ROOT" \
      --scene_name "$OBJ" \
      --output_dir "data/polyhaven_colmap/$OBJ" \
      --downsample 1
  else
    echo "[Step 1] COLMAP exists, skipping"
  fi

  # Step 2: LightSwitch relighting
  rm -rf relighting_outputs/rm_3_3/$OBJ/albedo* relighting_outputs/rm_3_3/$OBJ/orm*
  accelerate launch --num_processes 1 produce_gs_relightings.py \
    --dataset_type polyhaven \
    --data_root "$DATA_ROOT" \
    --relight_metadata "$METADATA_DIR/$OBJ.json" \
    --downsample 1

  # Step 3: Initial GS
  if [ ! -f "gs_outputs/$OBJ/chkpnt30000.pth" ]; then
    python gaussian-splatting/train.py \
      -s "data/polyhaven_colmap/$OBJ" \
      -m "gs_outputs/$OBJ" \
      --images images \
      --resolution 1 \
      --checkpoint_iterations 30000
  else
    echo "[Step 3] Initial GS exists, skipping"
  fi

  # Step 4: Fine-tune GS with relit images
  python gaussian-splatting/train.py \
    -s "relighting_outputs/rm_3_3/$OBJ/$RELIT" \
    --start_checkpoint "gs_outputs/$OBJ/chkpnt30000.pth" \
    --iterations 40000 \
    -m "gs_outputs/relit_gs/$OBJ/$RELIT" \
    --images images \
    --resolution 1 \
    --position_lr_init 0.0 \
    --position_lr_final 0.0 \
    --opacity_lr 0.0 \
    --scaling_lr 0.0 \
    --rotation_lr 0.0

  # Step 5: Render relit GS
  python gaussian-splatting/render.py \
    -m "gs_outputs/relit_gs/$OBJ/$RELIT"

  echo "=== Done: $OBJ → $RELIT ==="
done

echo ""
echo "All scenes complete."