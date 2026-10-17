<div align="center">

<h1>RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models</h1>

<p>
  <a href="https://arxiv.org/abs/2609.39801">
    <img src="https://img.shields.io/badge/arXiv-2609.39801-b31b1b.svg?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://github.com/steven-bao1/RATIO">
    <img src="https://img.shields.io/github/stars/steven-bao1/RATIO?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/steven-bao1/RATIO">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=steven-bao1.RATIO&amp;right_color=violet" alt="Visitors">
  </a>
</p>

<p>
  <a href="https://github.com/steven-bao1">Chengzhu Bao</a>,
  <a href="https://xianglongyan.github.io/">Xianglong Yan</a>,
  <a href="https://github.com/ZTA2785">Tianao Zhang</a>,
  <a href="https://github.com/ThrowBug">Jiaqi Chen</a>,
  Shaoqiu Zhang,
  <a href="http://yulunzhang.com/">Yulun Zhang</a>
</p>

</div>

## 🔥 News

- **2026-09-29:** The RATIO repository is launched.

> **Abstract:** Post-training quantization (PTQ) has become a widely adopted technique for reducing the memory footprint and inference cost of large language models (LLMs). However, recent studies reveal that when applied to reasoning models, PTQ not only degrades reasoning performance but also exacerbates overthinking, leading to longer reasoning trajectories. These issues may offset the efficiency gains expected from lower-precision inference. Existing approaches to mitigating these problems mainly rely on complex optimization procedures. More recent lightweight inference strategies instead use predefined overthinking markers, limiting their adaptability across quantized models. To address these issues, we propose **Reasoning Analysis and Token-level Inference Optimization (RATIO)**, a framework that identifies model-specific overthinking tokens and assigns each a tailored penalty. RATIO first introduces **Quantization-aware Reasoning Behavior Analysis (QRBA)** to identify overthinking tokens by analyzing discrepancies between full-precision and quantized models. It then adopts **Token-Specific Penalty Determination (TSPD)**, which leverages full-precision guidance to derive token-specific penalties without additional training. Extensive experiments show that RATIO achieves a better accuracy-efficiency trade-off than existing token-level interventions. Specifically, RATIO achieves up to **9.8 percentage points accuracy improvement** and reduces chain-of-thought (CoT) length by up to **51.3%** compared with quantized baselines. We will release all the code and implementation of RATIO.

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/steven-bao1/RATIO/main/figs/overview.png" alt="Overview of RATIO: reasoning behavior analysis, keyword selection, penalty determination, and calibrated inference">
</p>

## 🗂️ Contents

- [Installation](#installation)
- [Quantization](#quantization)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

<a id="installation"></a>

## 🔧 Installation

Requires an NVIDIA GPU.

```bash
git clone https://github.com/steven-bao1/RATIO.git
cd RATIO

conda create -n ratio python=3.12 -y
conda activate ratio

python -m pip install -r requirements.txt
bash scripts/install.sh
```

<a id="quantization"></a>

## Quantization

Run from the repository root and replace the model and calibration paths with your own.

### AWQ

```bash
CUDA_VISIBLE_DEVICES=0 python tools/quantize.py awq \
  --model /path/to/bf16-model \
  --output /path/to/awq-w3-model \
  --bits 3 --group-size 128 --seed 42 \
  --calibration pileval --awq-local-data /path/to/pileval
```

### GPTQ

```bash
CUDA_VISIBLE_DEVICES=0 python tools/quantize.py gptq \
  --model /path/to/bf16-model \
  --output /path/to/gptq-w3-model \
  --bits 3 --group-size 128 --seed 42 \
  --calibration wikitext2
```

Both entry points support weight bit widths from 2 to 8 via `--bits` (default: 3). These commands export weight-only fake-quantized checkpoints, not packed integer checkpoints.

Besides AWQ and GPTQ, `quantization/` includes QuaRot, KVQuant, SmoothQuant, and FlatQuant implementations.

<a id="data-preparation"></a>

## 📚 Data Preparation

Evaluation datasets are loaded from `./datasets`. Download them once:

```bash
# Optional HF mirror: export HF_ENDPOINT=https://hf-mirror.com
python tools/download_datasets.py --output-dir ./datasets --datasets all
```

<a id="usage"></a>

## 🚀 Usage

### 1. Collect experimental data

Edit `configs/collection.example.json` with your model and dataset paths, then install the recording hook:

```bash
python tools/install_recording_hook.py --check
python tools/install_recording_hook.py
```

<a id="direction-1-generated-prefix-evidence"></a>

Run both collection directions. Output directories are created automatically under `outputs/<BF16 model directory name>/`. The example below uses `DeepSeek-R1-Distill-Qwen-1.5B`; preparation prints the actual workspace path.

```bash
python tools/collect_data.py prepare \
  --config configs/collection.example.json

CUDA_VISIBLE_DEVICES=0 python tools/collect_data.py run \
  --workspace outputs/DeepSeek-R1-Distill-Qwen-1.5B/collection --stage all --skip-completed
```

`--stage all` runs the complete collection pipeline. When continuing a run, repeat the second command: `--skip-completed` verifies and skips completed stages. Incomplete or failed stages are not automatically reused.

### 2. Select keywords

```bash
python tools/select_keywords.py \
  --workspace outputs/DeepSeek-R1-Distill-Qwen-1.5B/collection
```

To reduce manual basic semantic checks, we use a lexicon compiled from historical reviews to filter clearly unsuitable tokens, such as single letters, variable-like fragments, and mathematical content words. This filter cannot remove every unsuitable token, so the remaining candidates still need to be checked. Candidates requiring contextual review are listed separately and must be reviewed against the original-response contexts before submission.

This produces preliminary keywords and candidates requiring contextual review. Review the candidates and their original-response contexts, then edit:

```text
outputs/DeepSeek-R1-Distill-Qwen-1.5B/selection/review/user_decisions.csv
```

Keep all rows and edit only `decision`: **leave blank to retain; enter `no` or `0` to exclude**. Complete the review before submitting; no written reason is required.

### 3. Submit the CSV and compute penalties

```bash
python tools/calibrate_penalties.py \
  --selection outputs/DeepSeek-R1-Distill-Qwen-1.5B/selection
```

The resulting `outputs/DeepSeek-R1-Distill-Qwen-1.5B/calibration/token_bias_map.json` can be used with `inference.py --token_bias_map`. Selection and calibration run on CPU and default to sibling directories of their inputs. Use `--output-dir` to specify an exact custom path, including when recalculating penalties or using models with the same directory name. Existing output directories are never overwritten.

### 4. Run inference with RATIO

Evaluate the corresponding quantized checkpoint with the generated penalty map (AWQ on GSM8K shown here):

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model /path/to/awq-w3-model \
  --dataset GSM8K \
  --token_bias_map outputs/DeepSeek-R1-Distill-Qwen-1.5B/calibration/token_bias_map.json \
  --output_dir outputs/DeepSeek-R1-Distill-Qwen-1.5B/evaluation/awq/gsm8k \
  --max_new_tokens 65536 --max_model_length 65536
```

Use the AWQ or GPTQ checkpoint corresponding to the model used for collection. Accuracy and mean generation length are saved in `summary.json`, with original responses in `responses.json`, under the specified evaluation directory.

<a id="code-structure"></a>

## 📂 Code Structure

```text
data_collection/                  # raw-data collection for both directions
keyword_selection/
├── __init__.py                   # package initialization
├── workflow.py                   # QRBA analysis workflow
├── inputs.py                     # input loading and validation
├── distributions.py              # QRBA aligned token distributions
├── stability.py                  # QRBA distribution stability analysis
├── semantic_filter.py            # post-QSTI lexical and token-shape exclusions
├── evidence.py                   # QRBA generated-prefix evidence
├── repetition.py                 # QRBA loop and repetition evidence
├── gates.py                      # QRBA candidate selection
├── decisions.py                  # user confirmation of candidate tokens
└── calibration.py                # TSPD token-specific penalty determination
penalty.py                        # inference-time logit adjustments
inference.py                      # model generation and evaluation
quantization/                     # quantization implementations
tools/                            # command-line entry points
scripts/                          # installation and workflow scripts
configs/                          # collection and selection configurations
```

<a id="results"></a>

## 📊 Results

RATIO reduces CoT length while generally preserving or improving reasoning accuracy across AIME, GPQA-Diamond, MATH-500, GSM8K, and HumanEval under AWQ-W3 and GPTQ-W3 quantization.

<details>
<summary>📊 Click to view the full experimental results</summary>

<p align="center">
  <img width="100%" src="https://raw.githubusercontent.com/steven-bao1/RATIO/main/figs/results.png" alt="Main results comparing BF16, quantized baselines, fixed penalties, and RATIO across five benchmarks">
</p>

</details>

<a id="citation"></a>

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{bao2026ratio,
  title         = {{RATIO}: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models},
  author        = {Chengzhu Bao and Xianglong Yan and Tianao Zhang and Jiaqi Chen and Shaoqiu Zhang and Yulun Zhang},
  year          = {2026},
  eprint        = {2609.39801},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.39801}
}
```

<a id="acknowledgements"></a>

## 💡 Acknowledgements

This work is released under the [Apache 2.0 License](https://github.com/XIANGLONGYAN/PT2-LLM/blob/main/LICENSE). Our execution and quantization framework builds on [Quantized-Reasoning-Models](https://github.com/ruikangliu/Quantized-Reasoning-Models). We also thank [vLLM](https://github.com/vllm-project/vllm) and [LightEval](https://github.com/huggingface/lighteval).
