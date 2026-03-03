cd /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LightSwitch

# 确保你有 Python 3.10
python -m venv lightswitch_env

# 激活虚拟环境
source lightswitch_env/bin/activate

ssh ubuntu@192.168.1.100
cd ~/LightSwitch
source lightswitch_env/bin/activate

# 1. SAM:

export OBJ=toycar
export IMAGES=images_4
export ENVMAP=aerodynamics_workshop
bash scripts/download_preprocess.sh
python scripts/generate_masks.py \
      --image_dir data/ceramic_vase_02_env_0/images \
      --initial_prompt 400 400 800 400 1000 300

accelerate launch produce_gs_relightings.py \
      --scene_dir data/ceramic_vase_02_env_0 \
      --image_dir_name images \
      --envmap_path data/light_probes/aerodynamics_workshop.hdr \
      --downsample 2

pip install git+https://github.com/facebookresearch/vggt
python3 scripts/vggt_colmap.py \
      --scene_dir data/ceramic_vase_02_env_0 \
      --image_dir_name images