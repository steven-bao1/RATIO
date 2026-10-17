from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset
import torch

from ..utils import utils
from ..utils import model_utils
from ..utils import data_utils


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.func = func

    def forward(self, *args, **kwargs):
        self.pre_rope_k = args[1].transpose(1, 2).detach().cpu()   # bsz, q_len, num_key_value_heads, head_dim
        bsz, q_len, num_key_value_heads, head_dim = self.pre_rope_k.shape
        self.pre_rope_k = self.pre_rope_k.reshape(-1, num_key_value_heads * head_dim)   # bsz*q_len, hidden_dim
        q, k = self.func(*args, **kwargs)
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    from ..utils import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)


@torch.no_grad()
def get_pre_rope_k_stats(args, model, tokenizer, dataset_name, num_samples=512, seq_len=512):
    model.eval()

    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config)

    if dataset_name == "pileval":
        dataset = load_dataset("json", data_files="./datasets/pile-val-backup/val.jsonl.zst", split="train")
        dataset = dataset.shuffle(seed=42)
    else:
        dataset = data_utils.get_loaders(dataset_name, nsamples=num_samples, seed=0, seqlen=seq_len, model=args.model, eval_mode=False)
    utils.distribute_model(model)
    device = next(model.parameters()).device

    pre_rope_k = defaultdict(list)
    for i in tqdm(range(num_samples)):
        if dataset_name == "pileval":
            input_ids = tokenizer(
                dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
            ).input_ids
        else:
            input_ids = dataset[i][0]
        model(input_ids.to(device))

        for name, module in model.named_modules():
            if isinstance(module, QKRotationWrapper):
                layer = int(name.split(".")[2])
                pre_rope_k[layer].append(module.pre_rope_k)

    pre_rope_k_scale_zero = []
    for layer in range(model.config.num_hidden_layers):
        x = torch.cat(pre_rope_k[layer], dim=0)
        if args.k_pre_bias:
            x = x - model.model.layers[layer].self_attn.k_proj.bias.to(x)
        scale, zero = get_scale_zero(x.transpose(0, 1), args.k_bits, not args.k_asym)
        pre_rope_k_scale_zero.append({
            "scale": scale,
            "zero": zero
        })

    return pre_rope_k_scale_zero


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max -1
    else:
        q_max, q_min = torch.tensor(2 ** bits - 1), 0
    return q_max, q_min


def get_scale_zero(x, bits, sym, clip_ratio=None):
    q_max, q_min = get_qmin_qmax(bits, sym)
    q_max = q_max.to(x)
    reshaped_x = x.reshape((-1, x.shape[-1]))
    xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
    tmp = torch.zeros_like(xmax)
    xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
    # # if self.groupsize > 0:
    # #     assert x.shape[-1] % self.groupsize == 0
    # #     x = x.reshape((-1, self.groupsize))
    # #     # TODO: add padding
    if clip_ratio is not None:
        xmax = xmax * clip_ratio
        xmin = xmin * clip_ratio
    if sym:
        xmax = torch.maximum(torch.abs(xmin), xmax)
        tmp = xmax == 0
        scale = (xmax / q_max)
        scale[tmp] = 1
        zero = torch.zeros_like(scale)
    else:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        scale = (xmax - xmin) / q_max
        zero = torch.round(-xmin / scale)

    return scale, zero
