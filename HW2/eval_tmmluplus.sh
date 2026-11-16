#!/bin/bash
# 用 HW1 的 eval_tmmluplus.py，在 HW1 的 TMMLU+ held-out set 上評測微調前後的模型（報告的 TMMLU+ 欄位）。
# 每個模型跑兩種設定：直接接題目（eval.json）與套 chat template（eval_template.json），
# 結果存在 results/tmmluplus/<模型名稱>/，最後印出各模型的 macro accuracy（HW1 的官方指標）。
if [ "$(basename "$PWD")" != "HW2" ]; then
    echo "錯誤：請在 HW2 資料夾下執行此腳本（目前位於：$PWD）" >&2
    exit 1
fi

source ../../miniconda3/etc/profile.d/conda.sh
conda activate hw2
# 任何一步失敗就停止；pipefail 讓 python ... | tee 的錯誤不會被 tee 吃掉
set -eo pipefail

HW2_DIR="$PWD"
HW1_DIR="$(cd ../HW1 && pwd)"
HOLDOUT="$(cd .. && pwd)/minimind/dataset/tmmluplus_local_eval_holdout.jsonl"
OUT_DIR="${HW2_DIR}/results/tmmluplus"

# 名稱=模型目錄（Hugging Face 格式）
MODELS=(
    "minimind-3=${HW2_DIR}/minimind-3"
    "E2=${HW2_DIR}/models/E2"
    "E3=${HW2_DIR}/models/E3"
)

# HW1 的 held-out set（seed 824、20% 的 validation，從未用於訓練）；不存在時用 HW1 相同的參數重新產生
if [ ! -f "${HOLDOUT}" ]; then
    echo "=== 找不到 ${HOLDOUT}，用 HW1 相同的參數重新產生 ==="
    cd "${HW1_DIR}"
    python prepare_tmmluplus_sft.py \
        --seed 824 \
        --holdout_frac 0.2 \
        --revision v1.1 \
        --holdout_out_path "${HOLDOUT}" \
        --out_path "$(dirname "${HOLDOUT}")/tmmluplus_sft.jsonl"
fi

mkdir -p "${OUT_DIR}"
echo "holdout: ${HOLDOUT}  md5: $(md5sum "${HOLDOUT}" | cut -d' ' -f1)" | tee "${OUT_DIR}/holdout_md5.txt"

cd "${HW1_DIR}"
for spec in "${MODELS[@]}"; do
    name="${spec%%=*}"
    model="${spec#*=}"
    out="${OUT_DIR}/${name}"
    mkdir -p "${out}"
    # 記錄實際評測的是哪一份權重，避免模型目錄被之後的訓練覆蓋後對不上
    {
        echo "evaluated_at: $(date -Iseconds)"
        echo "model_dir: ${model}"
        echo "model_md5: $(md5sum "${model}"/*.safetensors | cut -d' ' -f1)"
    } | tee "${out}/provenance.txt"

    python eval_tmmluplus.py eval \
        --model_path "${model}" \
        --data "${HOLDOUT}" \
        --save_predictions "${out}/p.jsonl" \
        --output "${out}/eval.json" 2>&1 | tee "${out}/eval.log"

    python eval_tmmluplus.py eval \
        --model_path "${model}" \
        --data "${HOLDOUT}" \
        --chat_template \
        --save_predictions "${out}/p_template.jsonl" \
        --output "${out}/eval_template.json" 2>&1 | tee "${out}/eval_template.log"
done

echo "=== TMMLU+ held-out macro accuracy ==="
python - "${OUT_DIR}" "${MODELS[@]}" <<'EOF'
import json, sys
out_dir, specs = sys.argv[1], sys.argv[2:]
print(f"{'model':<12}{'plain':>10}{'chat template':>16}")
for spec in specs:
    name = spec.split("=", 1)[0]
    acc = [json.load(open(f"{out_dir}/{name}/{f}.json"))["macro_accuracy"] for f in ("eval", "eval_template")]
    print(f"{name:<12}{acc[0]:>10.4f}{acc[1]:>16.4f}")
EOF
