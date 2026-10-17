# RATIO tokens and penalties

This document lists the configurations provided for reproducing the paper.

The tables below list the selected tokens and final penalty strengths for the
four models. Each model uses the same configuration for AWQ-W3 and GPTQ-W3.

`_` represents one leading space in the decoded token. Token IDs refer to each
model's original tokenizer; case and leading-space variants are distinct.
Penalties are positive values subtracted from the corresponding token logits
before temperature and top-p filtering: `logits[token_id] -= penalty`.

## DeepSeek-R1-Distill-Qwen-1.5B (21 tokens)

| Token ID | Token (`_` = leading space) | Penalty |
| ---: | --- | ---: |
| 714 | `_but` | 1.22 |
| 1128 | `_what` | 1.21 |
| 1430 | `_try` | 0.73 |
| 1549 | `_again` | 1.16 |
| 1632 | `_well` | 0.88 |
| 1988 | `_But` | 1.09 |
| 2494 | `_something` | 1.12 |
| 2578 | `_might` | 1.03 |
| 2753 | `No` | 0.74 |
| 3381 | `_thought` | 0.78 |
| 3983 | `But` | 1.09 |
| 4396 | `_correct` | 1.00 |
| 5978 | `_ensure` | 1.00 |
| 9454 | `Yes` | 0.90 |
| 11209 | `However` | 1.04 |
| 13824 | `_Wait` | 0.96 |
| 14190 | `Wait` | 1.12 |
| 31094 | `_conclude` | 1.02 |
| 44500 | `Thus` | 0.48 |
| 80022 | `Hmm` | 0.97 |
| 92014 | `Alternatively` | 0.97 |

## DeepSeek-R1-Distill-Qwen-7B (17 tokens)

| Token ID | Token (`_` = leading space) | Penalty |
| ---: | --- | ---: |
| 714 | `_but` | 1.06 |
| 1182 | `_back` | 0.58 |
| 1779 | `_check` | 0.48 |
| 1988 | `_But` | 1.15 |
| 2494 | `_something` | 1.09 |
| 2578 | `_might` | 0.75 |
| 3405 | `_question` | 1.32 |
| 3783 | `_wait` | 0.92 |
| 3983 | `But` | 1.00 |
| 4460 | `_trying` | 0.64 |
| 5978 | `_ensure` | 0.67 |
| 8088 | `but` | 1.04 |
| 13824 | `_Wait` | 1.18 |
| 14190 | `Wait` | 1.12 |
| 14301 | `_Thus` | 0.37 |
| 44500 | `Thus` | 0.36 |
| 80022 | `Hmm` | 1.03 |

## DeepSeek-R1-Distill-Llama-8B (17 tokens)

| Token ID | Token (`_` = leading space) | Penalty |
| ---: | --- | ---: |
| 1203 | `_back` | 1.00 |
| 1456 | `_try` | 0.90 |
| 1664 | `_well` | 0.86 |
| 1817 | `_check` | 0.93 |
| 2030 | `_But` | 1.12 |
| 2555 | `_something` | 1.34 |
| 3463 | `_thought` | 0.85 |
| 3488 | `_question` | 1.10 |
| 3604 | `_actually` | 0.95 |
| 3868 | `_wait` | 0.96 |
| 4071 | `But` | 1.07 |
| 4495 | `_correct` | 1.13 |
| 5084 | `_seems` | 1.11 |
| 7344 | `_maybe` | 0.93 |
| 8530 | `_perhaps` | 0.96 |
| 14144 | `_Wait` | 1.14 |
| 34431 | `something` | 1.29 |

## DeepSeek-R1-Distill-Qwen-14B (23 tokens)

| Token ID | Token (`_` = leading space) | Penalty |
| ---: | --- | ---: |
| 714 | `_but` | 1.04 |
| 1128 | `_what` | 1.08 |
| 1549 | `_again` | 0.79 |
| 1744 | `_think` | 0.74 |
| 1779 | `_check` | 0.53 |
| 1988 | `_But` | 1.28 |
| 2494 | `_something` | 1.63 |
| 3405 | `_question` | 0.83 |
| 3520 | `_actually` | 1.10 |
| 3783 | `_wait` | 0.96 |
| 3983 | `But` | 1.19 |
| 4396 | `_correct` | 0.95 |
| 4460 | `_trying` | 0.74 |
| 4977 | `_seems` | 0.92 |
| 7196 | `_maybe` | 0.80 |
| 7683 | `_confirm` | 0.72 |
| 8365 | `_perhaps` | 1.00 |
| 13824 | `_Wait` | 1.38 |
| 14190 | `Wait` | 1.37 |
| 33331 | `something` | 1.86 |
| 44500 | `Thus` | 0.38 |
| 80022 | `Hmm` | 1.29 |
| 88190 | `_Hmm` | 1.30 |
