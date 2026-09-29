# RATIO: Reasoning Analysis and Token-level Inference Optimization for Quantized Reasoning Models

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Coming%20Soon-red" alt="Paper coming soon">
  <a href="https://github.com/steven-bao1/RATIO">
    <img src="https://img.shields.io/github/stars/steven-bao1/RATIO?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/steven-bao1/RATIO">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=steven-bao1.RATIO&amp;right_color=violet" alt="Visitors">
  </a>
</p>

---

#### 🔥 News

- The method overview and main experimental results are available. Code and implementation details are coming soon!

---

> **Abstract:** Post-training quantization (PTQ) has become a widely adopted technique for reducing the memory footprint and inference cost of large language models (LLMs). However, recent studies reveal that when applied to reasoning models, PTQ not only degrades reasoning performance but also exacerbates overthinking, leading to longer reasoning trajectories. These issues may offset the efficiency gains expected from lower-precision inference. Existing approaches to mitigating these problems mainly rely on complex optimization procedures. More recent lightweight inference strategies instead use predefined overthinking markers, limiting their adaptability across quantized models. To address these issues, we propose **Reasoning Analysis and Token-level Inference Optimization (RATIO)**, a framework that identifies model-specific overthinking tokens and assigns each a tailored penalty. RATIO first introduces **Quantization-aware Reasoning Behavior Analysis (QRBA)** to identify overthinking tokens by analyzing discrepancies between full-precision and quantized models. It then adopts **Token-Specific Penalty Determination (TSPD)**, which leverages full-precision guidance to derive token-specific penalties without additional training. Extensive experiments show that RATIO achieves a better accuracy-efficiency trade-off than existing token-level interventions. Specifically, RATIO achieves up to **9.8 percentage points accuracy improvement** and reduces chain-of-thought (CoT) length by up to **51.3%** compared with quantized baselines. We will release all the code and implementation of RATIO.

<p align="center">
  <img width="100%" src="figs/overview.png" alt="Overview of RATIO: quantization-aware token identification and validation, token-specific penalty determination, and calibrated inference">
</p>

---

## ⚒️ TODO

- [x] Release the method overview and main experimental results
- [ ] Release QRBA token identification and validation code
- [ ] Release TSPD penalty calibration and inference code
- [ ] Release evaluation scripts and model-specific token configurations
- [ ] Add installation and usage instructions

## 🔗 Contents

- [Method](#method)
- [Models](#models)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

<a id="method"></a>

## 🧩 Method

RATIO is a **training-free** framework that reduces unnecessary reasoning in quantized models through model-specific token selection and token-specific logit penalties.

1. **Quantization-aware Reasoning Behavior Analysis (QRBA).** Quantization-Sensitive Token Identification (QSTI) compares full-precision and quantized models under identical reference prefixes to identify tokens with increased probability under both AWQ and GPTQ. Reasoning-context-aware Token Validation (RTV) then validates these candidates using actual quantized reasoning trajectories, combining probability shifts, associations with incorrect answers and repetitive reasoning, and contextual review.
2. **Token-Specific Penalty Determination (TSPD).** Using full-precision guidance, TSPD computes token-level log-odds corrections on fixed reference trajectories. It aggregates positive-shift events with a median for each quantizer, takes the smaller correction from AWQ and GPTQ, and normalizes by the median correction across the selected token set.
3. **Calibrated inference.** The resulting static penalties are subtracted from the logits of the selected tokens during generation. The full-precision model is used for offline analysis and calibration; inference uses the quantized model with the calibrated penalties.

<a id="models"></a>

## 🤖 Models

We evaluate RATIO on the following reasoning models:

| Model | Setting |
| --- | --- |
| DeepSeek-R1-Distill-Qwen-1.5B | BF16, AWQ-W3, GPTQ-W3 |
| DeepSeek-R1-Distill-Qwen-7B | BF16, AWQ-W3, GPTQ-W3 |
| DeepSeek-R1-Distill-Qwen-14B | BF16, AWQ-W3, GPTQ-W3 |
| DeepSeek-R1-Distill-Llama-8B | BF16, AWQ-W3, GPTQ-W3 |
| Qwen3-4B (thinking mode) | BF16, AWQ-W3, GPTQ-W3 |

Both AWQ and GPTQ use **3-bit weight quantization with a group size of 128**. Evaluation uses a temperature of **0.6**, top-p of **0.95**, and a maximum generation budget of **65,536 tokens**.

<a id="results"></a>

## 🔎 Results

We evaluate answer accuracy and CoT length on **AIME, GPQA-Diamond, MATH-500, GSM8K, and HumanEval**, comparing RATIO with uncalibrated quantized models and fixed-penalty decoding based on manually selected overthinking markers.

RATIO reduces average CoT length across all evaluated model–quantizer combinations while generally preserving or improving average accuracy. Highlights relative to the corresponding uncalibrated quantized baselines include:

- **Qwen-1.5B / AWQ-W3:** average accuracy increases from **30.88% to 40.66%** (+9.78 percentage points), while average CoT length decreases by **51.34%**.
- **Qwen-1.5B / GPTQ-W3:** average accuracy increases from **33.82% to 37.40%** (+3.58 percentage points), while average CoT length decreases by **25.66%**.
- **Qwen3-4B / GPTQ-W3:** average accuracy increases from **55.72% to 59.21%** (+3.49 percentage points), while average CoT length decreases by **17.28%**.

<details>
<summary>📊 Click to view the full experimental results</summary>

<p align="center">
  <img width="100%" src="figs/results.png" alt="Main results comparing accuracy and CoT length for BF16, AWQ-W3, and GPTQ-W3 models with fixed-penalty decoding and RATIO across five benchmarks">
</p>

</details>

<a id="citation"></a>

## Citation

The paper link and BibTeX citation will be added once available.

<a id="acknowledgements"></a>

## 💡 Acknowledgements

We thank the teams behind DeepSeek-R1, Qwen, Llama, AWQ, and GPTQ, as well as the creators of the evaluation benchmarks, for their contributions to open research.
