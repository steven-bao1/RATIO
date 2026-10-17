"""Preflight for the explicitly patched, pinned vLLM v0 sampler."""
import inspect


def require_recording_backend():
    import vllm.envs as envs
    from vllm.model_executor.layers import sampler
    if envs.VLLM_USE_V1 or envs.VLLM_USE_FLASHINFER_SAMPLER:
        raise RuntimeError("Collection requires VLLM_USE_V1=0 and VLLM_USE_FLASHINFER_SAMPLER=0")
    hook = getattr(sampler, "_record_online_post_top_p", None)
    if hook is None or "_record_online_post_top_p(" not in inspect.getsource(sampler.Sampler.forward):
        raise RuntimeError("Missing exact-nucleus hook. Run tools/install_recording_hook.py on the pinned vLLM source and reinstall it.")
    if "vllm_pre_intervention_sampling_logits" not in inspect.getsource(hook):
        raise RuntimeError("Installed recording hook does not support pre-forcing replay snapshots")
