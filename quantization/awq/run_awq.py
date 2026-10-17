
import os
import argparse
import json

import transformers
from .pre_quant import run_awq, apply_awq
from .quantizer import pseudo_quantize_model_weight


def save_awq_model(args):
    # load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cpu"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model.eval()

    # run awq
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=args.w_bits,
        q_config=args.q_config,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        calib_data=args.calib_data,
        model_path=args.model,
        calib_dataset_path=args.calib_dataset_path,
        calib_cache_path=args.calib_cache_path,
        calib_seed=args.calib_seed,
    )
    del model

    # save model
    if args.save_qmodel_path:
        os.makedirs(args.save_qmodel_path, exist_ok=True)
        tokenizer.save_pretrained(args.save_qmodel_path)
        # load model weights
        model_transformers = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="cpu"
        )
        apply_awq(model_transformers, awq_results)
        pseudo_quantize_model_weight(model_transformers, w_bit=args.w_bits, q_config={
            "zero_point": args.w_asym,
            "q_group_size": args.w_groupsize,
        })
        model_transformers.save_pretrained(args.save_qmodel_path)
        from .calib_data import LAST_CALIBRATION_AUDIT
        if LAST_CALIBRATION_AUDIT:
            with open(os.path.join(args.save_qmodel_path, "calibration_audit.json"), "w") as stream:
                json.dump(LAST_CALIBRATION_AUDIT, stream, indent=2)
        print(f"Model saved at {args.save_qmodel_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path of the hf model")
    # quantization config
    parser.add_argument("--w_bits", type=int, default=None)
    parser.add_argument("--w_groupsize", type=int, default=-1)
    parser.add_argument("--w_asym", action="store_true", help="asymmetric weight quantization")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--calib_data", type=str, default="pileval")
    parser.add_argument(
        "--calib_dataset_path",
        type=str,
        default=None,
        help="local calibration dataset path",
    )
    parser.add_argument(
        "--calib_cache_path",
        type=str,
        default=None,
        help="token cache for pile-uncopyrighted only",
    )
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument(
        "--save_qmodel_path", type=str, default=None, help="quantized model output directory"
    )
    args = parser.parse_args()

    # get quantization config (apart from w_bit)
    q_config = {
        "zero_point": args.w_asym,  # by default False
        "q_group_size": args.w_groupsize,  # whether to use group quantization
    }
    args.q_config = q_config
    print(args)

    save_awq_model(args)


if __name__ == "__main__":
    main()
