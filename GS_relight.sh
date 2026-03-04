export OBJ=pot_enamel_01_white_env_0
# ============================================================
# Step 1: 转换 polyhaven → COLMAP 格式（原始 scene）
# ============================================================
python scripts/polyhaven_to_colmap.py \
    --data_root /data/polyhaven_lvsm/test \
    --scene_name $OBJ \
    --output_dir data/polyhaven_colmap/ceramic_vase_02_white_env_0 \
    --downsample 1

# ============================================================
# Step 2: LightSwitch 重光照（生成 relit 图片 + COLMAP sparse）
# 注意：先删除旧的 albedo 缓存避免分辨率不匹配
# ============================================================
rm -rf relighting_outputs/rm_3_3/$OBJ/albedo*
rm -rf relighting_outputs/rm_3_3/$OBJ/orm*

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 produce_gs_relightings.py \
    --dataset_type polyhaven \
    --data_root /data/polyhaven_lvsm/test \
    --relight_metadata relight_metadata_smallsplit/$OBJ.json \
    --downsample 1

# ============================================================
# Step 3: 训练初始 GS（原始 scene）
# ============================================================
python gaussian-splatting/train.py \
    -s data/polyhaven_colmap/$OBJ \
    -m gs_outputs/$OBJ \
    --images images \
    --resolution 1 \
    --checkpoint_iterations 30000

# ============================================================
# Step 4: 冻结几何，用 relit 图片微调 GS 外观
# ============================================================
python gaussian-splatting/train.py \
    -s relighting_outputs/rm_3_3/$OBJ/$OBJ \
    --start_checkpoint gs_outputs/$OBJ/chkpnt30000.pth \
    --iterations 40000 \
    -m gs_outputs/relit_gs/$OBJ/$OBJ \
    --images images \
    --resolution 1 \
    --position_lr_init 0.0 \
    --position_lr_final 0.0 \
    --opacity_lr 0.0 \
    --scaling_lr 0.0 \
    --rotation_lr 0.0

# ============================================================
# Step 5: 渲染 relit GS
# ============================================================
python gaussian-splatting/render.py \
    -m gs_outputs/relit_gs/$OBJ/$OBJ