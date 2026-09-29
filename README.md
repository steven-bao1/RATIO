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

- **2026-09-29:** This repository is officially released!

---

> **Abstract:** Post-training quantization (PTQ) has become a widely adopted technique for reducing the memory footprint and inference cost of large language models (LLMs). However, recent studies reveal that when applied to reasoning models, PTQ not only degrades reasoning performance but also exacerbates overthinking, leading to longer reasoning trajectories. These issues may offset the efficiency gains expected from lower-precision inference. Existing approaches to mitigating these problems mainly rely on complex optimization procedures. More recent lightweight inference strategies instead use predefined overthinking markers, limiting their adaptability across quantized models. To address these issues, we propose **Reasoning Analysis and Token-level Inference Optimization (RATIO)**, a framework that identifies model-specific overthinking tokens and assigns each a tailored penalty. RATIO first introduces **Quantization-aware Reasoning Behavior Analysis (QRBA)** to identify overthinking tokens by analyzing discrepancies between full-precision and quantized models. It then adopts **Token-Specific Penalty Determination (TSPD)**, which leverages full-precision guidance to derive token-specific penalties without additional training. Extensive experiments show that RATIO achieves a better accuracy-efficiency trade-off than existing token-level interventions. Specifically, RATIO achieves up to **9.8 percentage points accuracy improvement** and reduces chain-of-thought (CoT) length by up to **51.3%** compared with quantized baselines. We will release all the code and implementation of RATIO.

<p align="center">
  <img width="100%" src="figs/overview.png" alt="Overview of RATIO: quantization-aware token identification and validation, token-specific penalty determination, and calibrated inference">
</p>

---

## ⚒️ TODO

- [ ] Complete this repository

## 🔗 Contents

- [ ] Quantization-aware Reasoning Behavior Analysis (QRBA)
- [ ] Token-Specific Penalty Determination (TSPD)
- [ ] [Results](#results)
- [ ] [Citation](#citation)
- [ ] [Acknowledgements](#acknowledgements)

<a id="results"></a>

## 🔎 Results

RATIO reduces CoT length while generally preserving or improving reasoning accuracy across AIME, GPQA-Diamond, MATH-500, GSM8K, and HumanEval under AWQ-W3 and GPTQ-W3 quantization.

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
