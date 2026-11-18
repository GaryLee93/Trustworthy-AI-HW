#!/bin/bash
# Reproduce the final model (Experiment 3: minimind-3 + tool-call & reading SFT) from the data in data_manifest.md.
#
# Submission layout (this file is <student_id>/code/run.sh):
#   <student_id>/model/  <student_id>/code/  <student_id>/data_manifest.md  <student_id>/logs/  <student_id>/report.pdf
#
# Usage (from anywhere):
#   bash code/run.sh [OUTPUT_MODEL_DIR]        # default: <student_id>/model_reproduced
# The submitted model/ is never overwritten. Environment variables:
#   WORK_DIR  where minimind-3 and the MiniMind repo are downloaded (default: <student_id>/work)
#   LOG_DIR   where the logs are written                            (default: <student_id>/logs_reproduced)
# Requirements: Python 3.10+, network access to huggingface.co and github.com; a CUDA GPU is recommended.
set -eo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${CODE_DIR}")"
OUT_DIR="$(mkdir -p "${1:-${ROOT_DIR}/model_reproduced}" && cd "${1:-${ROOT_DIR}/model_reproduced}" && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs_reproduced}"
mkdir -p "${WORK_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"

# Pinned inputs (see data_manifest.md)
MINIMIND_COMMIT=233d00fa068ec2b8c0d4f07b703934d6391a5a73   # MiniMind repo: trainer utilities used by train_sft.py
BASE_MODEL_MD5=15c4f9960b1a1ee82ff7e3615b5f668c             # model.safetensors of jingyaogong/minimind-3 we started from
SFT_MD5S="b76102ec5c17c0902bd97c2013eecfea  sft_data/sft_tool.jsonl
03fbcce7752b503e03b1049941c719fe  sft_data/sft_reading.jsonl
5dce501877fcada774ad63b8671ba73b  sft_data/sft_mix.jsonl"

echo "=== 1. Python dependencies ==="
python -c "import torch, transformers, numpy, huggingface_hub" 2>/dev/null || pip install torch transformers numpy huggingface_hub
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, \
'| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" | tee "${LOG_DIR}/environment.txt"

echo "=== 2. Official HW2 data (regenerated if missing, then checked against MD5SUMS_public) ==="
[ -f data/train/memory.jsonl ] || python gen_hw2_data.py --split train --seed 1 --n_personas 300 --n_tool 3000 --out data/train
[ -f data/val/memory.jsonl ]   || python gen_hw2_data.py --split val   --seed 2 --n_personas 30  --n_tool 200  --out data/val
md5sum -c MD5SUMS_public

echo "=== 3. SFT data: sft_tool.jsonl + sft_reading.jsonl -> sft_mix.jsonl ==="
python build_sft_data.py --data_dir data/train --out_dir sft_data
md5sum sft_data/*.jsonl | tee "${LOG_DIR}/sft_data_md5.txt"
echo "${SFT_MD5S}" | md5sum -c -

echo "=== 4. Base model (jingyaogong/minimind-3) and MiniMind trainer code ==="
python - "${WORK_DIR}/minimind-3" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("jingyaogong/minimind-3", local_dir=sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.jinja"])
EOF
if [ "$(md5sum "${WORK_DIR}/minimind-3/model.safetensors" | cut -d' ' -f1)" != "${BASE_MODEL_MD5}" ]; then
    echo "WARNING: minimind-3 on Hugging Face differs from the version used in the report (md5 ${BASE_MODEL_MD5})." >&2
fi
if [ ! -d "${WORK_DIR}/minimind/.git" ]; then
    git clone https://github.com/jingyaogong/minimind.git "${WORK_DIR}/minimind"
fi
git -C "${WORK_DIR}/minimind" checkout -q "${MINIMIND_COMMIT}"

echo "=== 5. SFT (Experiment 3 settings) -> ${OUT_DIR} ==="
start=${SECONDS}
MINIMIND_DIR="${WORK_DIR}/minimind" python train_sft.py \
    --model_dir "${WORK_DIR}/minimind-3" \
    --data_path sft_data/sft_mix.jsonl \
    --save_dir "${OUT_DIR}" \
    --epochs 1 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_seq_len 768 \
    --log_interval 250 \
    --save_interval 500 \
    --deterministic \
    --seed 824 2>&1 | tee "${LOG_DIR}/E3_train.log"
echo "training time: $((SECONDS - start)) s" | tee -a "${LOG_DIR}/E3_train.log"

echo "=== 6. Evaluation on the validation set ==="
python eval_agent.py \
    --model_dir "${OUT_DIR}" \
    --agent my_agent:MyAgent \
    --data_dir data/val \
    --out "${LOG_DIR}/eval_val" 2>&1 | tee "${LOG_DIR}/eval_val.log"

echo "Done. Model: ${OUT_DIR}   Logs: ${LOG_DIR}"
