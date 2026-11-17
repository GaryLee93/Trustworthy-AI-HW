# HW2 Data Manifest

All training data come from the public HW2 training split. No validation data, no test data, and no LLM-generated data were used for training.

| File | Used for | Samples | MD5 |
|---|---|---|---|
| `data/train/tool.jsonl` | source of `sft_tool.jsonl` | 3,000 | `2835d861ceebc3e6f5546a37aae3b0cd` |
| `data/train/memory.jsonl` | source of `sft_reading.jsonl` | 300 personas / 3,000 questions | `ae69dc49567b88c49f1f36d74e214964` |
| `sft_data/sft_tool.jsonl` | Experiment 2 training | 3,000 | `b76102ec5c17c0902bd97c2013eecfea` |
| `sft_data/sft_reading.jsonl` | part of `sft_mix.jsonl` | 3,000 | `03fbcce7752b503e03b1049941c719fe` |
| `sft_data/sft_mix.jsonl` | Experiment 3 training | 6,000 | `5dce501877fcada774ad63b8671ba73b` |
| `data/val/memory.jsonl`, `data/val/tool.jsonl` | evaluation only | 300 questions / 200 items | `7c1937fd6a6b6cdc432b45e5a0b39185`, `763e3544a115c66d2edbb8772ff48e65` |
| `minimind/dataset/tmmluplus_local_eval_holdout.jsonl` | TMMLU+ evaluation only | 437 | `4a797681a390bc9d685222f379ce1df5` |

Paths of the HW2 files are relative to `code/`.

## data/train/tool.jsonl and data/train/memory.jsonl
### Basic Info
- **Source**: the official `hw2_student_release.zip`, used as-is
- **Generator**: `gen_hw2_data.py --split train --seed 1 --n_personas 300 --n_tool 3000` with `templates_public.py`; the MD5s match `MD5SUMS_public`
- **License**: course material
- **Language**: Traditional Chinese
- **Content**:
  - `memory.jsonl`: 300 synthetic personas, each with 8 sessions (128 utterances) and 10 four-option questions; `annotations` mark facts, updates and statements about other people
  - `tool.jsonl`: 3,000 short contexts with the expected action (`memory_save`, `memory_search`, `calculator`, `get_date`, or no tool call)

### Verification Command
```bash
cd code
md5sum -c MD5SUMS_public
```

## sft_tool.jsonl
### Basic Info
- **Source**: `data/train/tool.jsonl`
- **Used by**: Experiment 2 (alone) and Experiment 3 (inside `sft_mix.jsonl`)
- **Format**: one JSON object per line: `{"task": "tool", "id", "kind", "messages", "tools", "target"}`

| Kind | Target | Samples |
|---|---|---|
| save, update | `memory_save` | 885 + 287 |
| search, search_calc | `memory_search` | 604 + 167 |
| calculator | `calculator` | 378 |
| date | `get_date` | 179 |
| answer_noise, answer_other | no tool call | 326 + 174 |

### Data Processing
1. **Prompt**: `build_sft_data.py` calls `MyAgent.next_action(context)` with `generate()` intercepted, and stores the exact `messages` (system prompt + the utterances joined by `\n`) and `tools` (the four tools in `tool_spec.json`) that the evaluation sends to the model.
2. **Target**:
   - when a tool should be called: the expected call in the chat-template format, e.g. `<tool_call>\n{"name": "calculator", "arguments": {"expression": "473+336"}}\n</tool_call>`. Every target passes the official `parse_call` and `args_match` of `eval_agent.py`.
   - otherwise: one of three fixed replies (`好的，我知道了。`, `嗯嗯，了解！`, `謝謝你跟我分享。`), chosen with `random.Random(0)`.

## sft_reading.jsonl
### Basic Info
- **Source**: `data/train/memory.jsonl`
- **Used by**: Experiment 3 (inside `sft_mix.jsonl`)
- **Format**: one JSON object per line: `{"task": "reading", "id", "type", "context", "question", "target"}`
- **Composition**: single 600, update 462, multi 442, abstain 445, distractor 451, tool 600

### Data Processing
1. **Context**: for each persona, `build_sft_data.py` calls `MyAgent` in the same order as `eval_agent.py` (`reset` → `new_session` / `chat` for every utterance → `save_snapshot` → for each question `load_snapshot` → `prepare(question)`), and keeps the last 300 characters of the output (`--max_context_chars`, the evaluation default).
   The context therefore depends on `my_agent.py`; the released file was built with Experiment 1 - Update 4.
2. **Question**: the question text and the four options.
3. **Target**: the text of the correct option (not the letter), since the evaluation scores the option text.

Example:
```json
{"task": "reading", "id": "trp0001_q01", "type": "single",
 "context": "房東漲房租了，現在一個月15000元\n我現在比較常打桌球，以前的運動不玩了\n我家在基隆",
 "question": {"question": "我現在住在哪個城市？", "options": {"A": "你沒有提過", "B": "桃園", "C": "台南", "D": "基隆"}},
 "target": "基隆"}
```

## sft_mix.jsonl
- `sft_tool.jsonl` followed by `sft_reading.jsonl` (6,000 samples); `train_sft.py` shuffles the samples every epoch.

## How the SFT Data Is Used in Training
`train_sft.py` turns each sample into (prompt, target) and computes the loss on the target only:
- **tool**: prompt = `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True)`, the same as `BaseAgent.generate()`; target = `target` + `<|im_end|>`.
- **reading**: prompt = `eval_agent.build_prompt(tokenizer, context, question, use_chat=True)`, which ends with `答案：`; target = option text. The prompt and the target are tokenized separately, the same as `eval_agent.score_options()`.

| Setting | Experiment 2 | Experiment 3 |
|---|---|---|
| Data | `sft_tool.jsonl` | `sft_mix.jsonl` |
| Start model | [`jingyaogong/minimind-3`](https://huggingface.co/jingyaogong/minimind-3) (63.9M, `model.safetensors` MD5 `15c4f9960b1a1ee82ff7e3615b5f668c`) | same |
| Epochs / steps | 1 / 188 | 1 / 375 |
| Batch size / learning rate | 16 / 1e-5 (MiniMind cosine schedule) | same |
| Precision / seed | bf16 / 824, `--deterministic` | same |
| Sequence length (tokens) | max 700, mean 645 | max 700, mean 389 |

Training logs: `logs/E2_train.log`, `logs/E3_train.log`; data MD5s at training time: `logs/E3_data_md5.txt`.

## Evaluation Data (never used for training)
- **HW2 validation set** (`data/val`, `gen_hw2_data.py --split val --seed 2 --n_personas 30 --n_tool 200`): used only by `eval_agent.py` and `analyze_context.py`.
  Because the training and validation sets share the public templates, 49 of the 300 validation memory prompts (context, question and options) are identical to prompts in `sft_reading.jsonl`, and 42 of the 200 validation tool contexts appear in `data/train/tool.jsonl`. Validation scores therefore overestimate the test performance.
- **TMMLU+ held-out set** (`tmmluplus_local_eval_holdout.jsonl`, 437 questions): the HW1 local holdout, i.e. 20% of the [`ikala/tmmluplus`](https://huggingface.co/datasets/ikala/tmmluplus) validation split (revision `v1.1`, stratified by subject, seed 824), written by `HW1/prepare_tmmluplus_sft.py`. It was used only to measure TMMLU+ before and after fine-tuning (`eval_tmmluplus.sh`).

## LLM-Generated Data
No LLM-generated data were used for training or for any score in the report.
During the development of Experiment 1, small sets of paraphrased sentences and questions written by an LLM were used only to check whether the rule-based parts of `my_agent.py` generalize to unseen phrasings; they were not saved and are not part of any dataset above.

## Generation Commands
`code/run.sh` reproduces the final model (Experiment 3) from the data above: it checks the official data with `MD5SUMS_public` (regenerating it with `gen_hw2_data.py` if missing), builds the SFT data and checks the MD5s in the table above, downloads minimind-3 and the MiniMind trainer utilities (commit `233d00f`), runs `train_sft.py`, and evaluates the model on the validation set.
```bash
bash code/run.sh            # model -> model_reproduced/, logs -> logs_reproduced/
```
The SFT data alone can be rebuilt with:
```bash
cd code
md5sum -c MD5SUMS_public
python build_sft_data.py --data_dir data/train --out_dir sft_data
md5sum sft_data/*.jsonl
```
