# Meta fixed-50 overthinking-marker baseline

Source: Lotfi et al., [Quantized Reasoning Models Think They Need to Think
Longer, but They Do Not](https://arxiv.org/abs/2606.00206).
This is a separate baseline, not RATIO's selected keyword set. The public code
already implements it in `penalty.py:FixedMarkerPenalty`.

## Candidate strengths

Appendix A.3 gives the positive sweep:

`0.5, 0.7, 1.0, 1.5, 2.0, 2.6, 3.0, 4.0`.

Use `0` for the unmodified base. The CLI accepts other finite nonnegative
strengths too; these eight values are paper reference candidates, not promises
of the best score on a particular model. The paper also studies a negative
boosting sweep, which this fixed-penalty CLI intentionally does not support.

## Marker list

Table 8's 50 literal strings are stored in `penalty.py:FIXED_MARKERS`. Here `_`
denotes a leading space, not a literal underscore:

```text
_perhaps _maybe _wait _Wait _actually _hold _Hmm _hmm
_Alternatively _alternatively _However _however _instead _Instead
_But _but _though _although _yet _rather _unless _otherwise
_nonetheless _nevertheless _regardless _still _anyway _Or _or
_either _whether _uncertain _unsure _possibly _might _could
_another _different _reconsider _rethink _backtrack _retry
_recheck _revisit _doubt _confused _wrong _mistake _error _incorrect
```

The strings are encoded with the inference tokenizer, keeping case and leading
spaces. They are not necessarily 50 distinct single-token IDs. A multi-token
marker applies the penalty to its last token when the generated prefix matches
its preceding tokens; prompt tokens do not trigger that match. Duplicate target
IDs are penalized only once per step. The matching token's raw logit is reduced
by lambda **before temperature and top-p**. No median normalization occurs.
The fixed option and a RATIO token-bias map cannot be combined.

## Run

From the project root, for one task:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model /path/to/YOUR-QUANTIZED-MODEL \
  --dataset MATH-500 --fixed_penalty 1.5 \
  --output_dir outputs/fixed1p5/math500
```

For the five-task suite, first run the preflight:

```bash
python tools/run_suite.py --dry-run \
  --model /path/to/YOUR-QUANTIZED-MODEL \
  --fixed_penalty 1.5 --output-root outputs/fixed1p5 \
  --allow-code-execution
```

Then remove `--dry-run` **only in an appropriate external execution sandbox**:
HumanEval executes untrusted generated code. `--allow-code-execution` is consent,
not isolation. Repeat with each chosen strength and a unique output directory;
select dataset/generation settings explicitly when reproducing a paper run.
No GPU sweep starts merely by reading this configuration or invoking `--help`.
