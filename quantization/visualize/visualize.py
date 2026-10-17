import os
import argparse
from tqdm import tqdm

import torch
import transformers

from ..utils import utils
from ..utils import data_utils
from ..utils import model_utils
from . import get_acts_utils
from . import tsne_utils
from . import plot_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='user-selected reasoning model path')
    parser.add_argument('--exp', type=str, default="dataset-domain-gap", choices=["dataset-domain-gap", "kcache-bias", "massive-acts"])
    args = parser.parse_args()

    args.model_name = args.model.split("/")[-1]
    args.output_path = os.path.join("./outputs/visualize", args.model_name, args.exp)
    os.makedirs(args.output_path, exist_ok=True)

    return args


@torch.no_grad()
def main(args):
    model = model_utils.get_model(args.model)
    num_layers = model.config.num_hidden_layers

    if args.exp == "dataset-domain-gap":    # visualize activations from pre-training and self-generated reasoning dataset (10K tokens)
        dataset_names = ["wikitext2", "reasoning-numina-math-1.5", "reasoning-livecodebench"]
        label_names = ["Wikitext2", "Numina-Math-1.5", "LiveCodeBench"]

        utils.distribute_model(model)
        datasets = [data_utils.get_loaders(dataset_name, nsamples=128, seqlen=2048, model=args.model, eval_mode=False) for dataset_name in dataset_names]
        linear_acts_list = [get_acts_utils.get_acts(model, dataset, num_tokens=10240, type="linear") for dataset in datasets]
        del model
        utils.cleanup_memory()

        names = linear_acts_list[0].keys()
        for name in tqdm(names):
            layer_idx = int(name.split(".")[2])
            if layer_idx in range(0, num_layers, num_layers // 4):
                # t-SNE visualization (randomly sample 128 tokens)
                handles_tsne, labels_tsne = tsne_utils.tsne_visualization([linear_acts[name].float().numpy() for linear_acts in linear_acts_list],
                                                                            f"{args.output_path}/tsne", name, label_names,
                                                                            max_samples_per_batch=128)

                # 2D plot
                flatnesses = [torch.mean(linear_acts[name], dim=0).float().numpy() for linear_acts in linear_acts_list]
                handles_flatness, labels_flatness = plot_utils.plot_flatness(f"{args.output_path}/flatness", name, flatnesses, label_names)

                # # activation visualization
                # for label_name, linear_acts in zip(label_names, linear_acts_list):
                #     plot_utils.plot_activations(linear_acts[name].float().numpy(), 128, name, f"{args.output_path}/acts/{label_name}")
        plot_utils.save_legend_as_pdf(handles_tsne, labels_tsne, f"{args.output_path}/tsne")
        plot_utils.save_legend_as_pdf(handles_flatness, labels_flatness, f"{args.output_path}/flatness")
    elif args.exp == "kcache-bias":     # visualize the impact of bias on K cache quantization
        assert isinstance(model, transformers.Qwen2ForCausalLM), "K cache bias issues only exist in Qwen 1.5B and 7B models"
        dataset_name = "reasoning-numina-math-1.5"

        # before and after bias (bias introduces extreme outlier channels)
        dataset = data_utils.get_loaders(dataset_name, nsamples=128, seqlen=2048, model=args.model, eval_mode=False)
        kcache_before_bias, kcache_before_rope, kcache = get_acts_utils.get_kcache(model, dataset, num_tokens=32768)
        del model
        utils.cleanup_memory()

        names = kcache_before_bias.keys()
        for name in tqdm(names):
            layer_idx = int(name.split(".")[2])
            if layer_idx in range(0, num_layers, num_layers // 4):
                plot_utils.plot_activations(kcache_before_bias[name].float().numpy(), 128, f"layers.{layer_idx}.kcache_pre_bias", f"{args.output_path}/outlier_channel")
                plot_utils.plot_activations(kcache[name].float().numpy(), 128, f"layers.{layer_idx}.kcache", f"{args.output_path}/outlier_channel")

        # before and after RoPE (after RoPE, bias introduces channel distribution variantions along sequence length)
        model = model_utils.get_model(args.model)
        dataset = data_utils.get_loaders(dataset_name, nsamples=128, seqlen=8192, model=args.model, eval_mode=False)
        _, kcache_before_rope, kcache = get_acts_utils.get_kcache(model, dataset, num_tokens=32768, keep_seqlen_dim=True)
        del model
        utils.cleanup_memory()

        step_size = 2048
        kcache_list, kcache_before_rope_list = [], []
        label_names = ["0~2K", "2K~4K", "4K~6K", "6K~8K"]
        for i in range(0, 8192, step_size):
            kcache_list.append({})
            kcache_before_rope_list.append({})
            for (name, k), (name, k_before_rope) in zip(kcache.items(), kcache_before_rope.items()):
                hidden_dim = k.shape[-1]
                kcache_list[-1][name] = k[:, i:i+step_size, :].reshape(-1, hidden_dim)
                kcache_before_rope_list[-1][name] = k_before_rope[:, i:i+step_size, :].reshape(-1, hidden_dim)

        names = kcache_list[0].keys()
        for name in tqdm(names):
            layer_idx = int(name.split(".")[2])
            if layer_idx in range(0, num_layers, num_layers // 4):
                flatnesses = [torch.mean(kcache[name], dim=0).float().numpy() for kcache in kcache_list]
                handles_flatness, labels_flatness = plot_utils.plot_flatness(f"{args.output_path}/seq-flatness", f"model.layers.{layer_idx}.kcache", flatnesses, label_names)
                flatnesses = [torch.mean(kcache_before_rope[name], dim=0).float().numpy() for kcache_before_rope in kcache_before_rope_list]
                handles_flatness, labels_flatness = plot_utils.plot_flatness(f"{args.output_path}/seq-flatness", f"model.layers.{layer_idx}.kcache_before_rope", flatnesses, label_names)
        plot_utils.save_legend_as_pdf(handles_flatness, labels_flatness, f"{args.output_path}/seq-flatness")
    elif args.exp == "massive-acts":    # visualize massive activations from reasoning models (30K tokens)
        utils.distribute_model(model)
        dataset = data_utils.get_loaders("reasoning-numina-math-1.5", nsamples=128, seqlen=2048, model=args.model, eval_mode=False)
        block_acts = get_acts_utils.get_acts(model, dataset, num_tokens=32768, type="block")
        attns = get_acts_utils.get_attn(model, dataset, num_tokens=64)
        del model
        utils.cleanup_memory()

        # layerwise top activation magnitudes
        stats_list = []
        for layer in range(num_layers):
            stats = {
                "Top 1": block_acts[f"model.layers.{layer}"].max().item(),
                "Median": torch.median(block_acts[f"model.layers.{layer}"]).item()
            }
            stats_list.append(stats)
        plot_utils.plot_massive_activations_stats(stats_list, f"{args.output_path}/stats")
   
        # attention sink
        for layer in range(0, num_layers, num_layers // 4):
            plot_utils.plot_attn_map(attns[layer], f"model.layers.{layer}-attns", f"{args.output_path}/attn")

        # massive activations
        names = block_acts.keys()
        for name in tqdm(names):
            layer_idx = int(name.split(".")[2])
            if layer_idx in range(0, num_layers, num_layers // 4):
                plot_utils.plot_activations(block_acts[name].float().numpy(), 128, f"{name}", f"{args.output_path}/acts")


if __name__ == '__main__':
    args = parse_args()
    main(args)
