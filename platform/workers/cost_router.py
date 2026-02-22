GPU_MODALITIES = {
    "cassi_sci", "fpm", "ptychography", "holography",
    "light_sheet", "neural_field", "nerf", "3dgs",
}
CPU_MODALITIES = {
    "ct_diagnostic", "mri_parallel", "spectral_simple",
}


def should_use_gpu(spec: dict, compute_mode: str, tensor_size_mb: float = 0) -> bool:
    if compute_mode == "cpu":
        return False
    if compute_mode == "gpu":
        return True
    modality = spec.get("states", {}).get("physics", {}).get("modality", "")
    if modality in CPU_MODALITIES:
        return False
    if modality in GPU_MODALITIES or tensor_size_mb > 500:
        return True
    return False


def estimate_cost(spec: dict, use_gpu: bool) -> dict:
    if use_gpu:
        return {"mode": "GPU (Modal T4)", "est_time_s": "30-120", "est_cost_usd": "~$0.01-0.05"}
    return {"mode": "CPU", "est_time_s": "5-30", "est_cost_usd": "$0.00"}
