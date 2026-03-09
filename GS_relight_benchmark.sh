#!/bin/bash
# Run GS_relight pipeline on 1 scene and record:
#   1. End-to-end run time
#   2. Peak GPU and CPU consumption
#   3. Per-step inference time (Steps 1, 2, 3, 4, 5)
#
# Usage: bash GS_relight_benchmark.sh [scene_name]
#   Default scene: ceramic_vase_02_white_env_0

set -e

OBJ="${1:-cardboard_box_01_env_1}"
DATA_ROOT="/data/polyhaven_lvsm/test"
METADATA_DIR="data_samples/relight_metadata"
export CUDA_VISIBLE_DEVICES=0

BENCH_DIR="benchmark_${OBJ}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_DIR"
LOG="$BENCH_DIR/benchmark.log"
GPU_LOG="$BENCH_DIR/gpu.log"
CPU_LOG="$BENCH_DIR/cpu.log"

echo "Benchmark: $OBJ" | tee "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Start monitors in background ---
monitor_gpu() {
  while true; do
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> "$GPU_LOG"
    sleep 2
  done
}
monitor_cpu() {
  while true; do
    if grep -q 'cpu ' /proc/stat 2>/dev/null; then
      grep 'cpu ' /proc/stat | head -1 >> "$CPU_LOG"
    fi
    sleep 2
  done
}

monitor_gpu &
GPU_PID=$!
monitor_cpu &
CPU_PID=$!
trap "kill $GPU_PID $CPU_PID 2>/dev/null" EXIT

# --- Timing helper ---
step_start() { STEP_START=$(date +%s.%N); }
step_end() {
  local step_name="$1"
  local end=$(date +%s.%N)
  local dur=$(echo "$end - $STEP_START" | bc 2>/dev/null || echo "N/A")
  echo "[$step_name] ${dur}s"
  echo "[$step_name] ${dur}s" >> "$LOG"
  eval "STEP_${step_name// /_}_DUR=$dur"
}

# --- Main ---
TOTAL_START=$(date +%s.%N)

if [ ! -f "$METADATA_DIR/$OBJ.json" ]; then
  echo "Error: metadata not found: $METADATA_DIR/$OBJ.json"
  exit 1
fi

RELIT=$(python3 -c "import json; print(json.load(open('$METADATA_DIR/$OBJ.json'))['relit_scene_name'])")

# Step 1
step_start
if [ ! -f "data/polyhaven_colmap/$OBJ/sparse/0/cameras.bin" ]; then
  python scripts/polyhaven_to_colmap.py \
    --data_root "$DATA_ROOT" \
    --scene_name "$OBJ" \
    --output_dir "data/polyhaven_colmap/$OBJ" \
    --downsample 1
else
  echo "[Step 1] COLMAP exists, skipping"
fi
step_end "Step1_polyhaven2colmap"

# Step 2
step_start
rm -rf relighting_outputs/rm_3_3/$OBJ/albedo* relighting_outputs/rm_3_3/$OBJ/orm*
accelerate launch --num_processes 1 produce_gs_relightings.py \
  --dataset_type polyhaven \
  --data_root "$DATA_ROOT" \
  --relight_metadata "$METADATA_DIR/$OBJ.json" \
  --downsample 1
step_end "Step2_relighting"

# Step 3
step_start
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
step_end "Step3_initial_GS"

# Step 4
step_start
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
step_end "Step4_relit_GS_finetune"

# Step 5
step_start
python gaussian-splatting/render.py \
  -m "gs_outputs/relit_gs/$OBJ/$RELIT"
step_end "Step5_render"

TOTAL_END=$(date +%s.%N)
TOTAL_DUR=$(echo "$TOTAL_END - $TOTAL_START" | bc 2>/dev/null || echo "N/A")

# --- Stop monitors ---
kill $GPU_PID $CPU_PID 2>/dev/null || true
trap - EXIT

# --- Compute peak GPU ---
PEAK_GPU_MB=0
PEAK_GPU_UTIL=0
if [ -f "$GPU_LOG" ]; then
  while IFS= read -r line; do
    mem=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    util=$(echo "$line" | cut -d',' -f3 | tr -d ' ')
    [[ "$mem" =~ ^[0-9]+$ ]] && [ "$mem" -gt "$PEAK_GPU_MB" ] && PEAK_GPU_MB=$mem
    [[ "$util" =~ ^[0-9]+$ ]] && [ "$util" -gt "$PEAK_GPU_UTIL" ] && PEAK_GPU_UTIL=$util
  done < "$GPU_LOG"
fi

# --- Compute CPU from /proc/stat samples ---
PEAK_CPU_PCT="N/A"
if [ -f "$CPU_LOG" ] && [ -n "$(command -v python3)" ]; then
  PEAK_CPU_PCT=$(python3 - "$CPU_LOG" << 'PYEOF'
import sys
try:
    path = sys.argv[1]
    samples = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "cpu":
                nums = [int(x) for x in parts[1:8]]
                total = sum(nums)
                idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
                used = total - idle
                samples.append(100.0 * used / total if total else 0)
    print("{:.1f}".format(max(samples))) if samples else print("N/A")
except Exception:
    print("N/A")
PYEOF
) || PEAK_CPU_PCT="N/A"
fi

# --- Summary ---
echo "" | tee -a "$LOG"
echo "========== BENCHMARK SUMMARY ==========" | tee -a "$LOG"
echo "Scene: $OBJ" | tee -a "$LOG"
echo "Relit: $RELIT" | tee -a "$LOG"
echo "1. End-to-end run time: ${TOTAL_DUR}s" | tee -a "$LOG"
echo "2. Peak GPU memory: ${PEAK_GPU_MB} MiB, Peak GPU util: ${PEAK_GPU_UTIL}%" | tee -a "$LOG"
echo "   Peak CPU usage: ${PEAK_CPU_PCT}%" | tee -a "$LOG"
echo "3. Per-step times:" | tee -a "$LOG"
grep '^\[Step' "$LOG" | tee -a "$LOG"
echo "Output: $BENCH_DIR" | tee -a "$LOG"
