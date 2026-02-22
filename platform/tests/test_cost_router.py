from workers.cost_router import should_use_gpu, estimate_cost

def test_cpu_mode_never_gpu():
    assert should_use_gpu({"states": {"physics": {"modality": "cassi_sci"}}}, "cpu") is False

def test_gpu_mode_always_gpu():
    assert should_use_gpu({}, "gpu") is True

def test_auto_gpu_modality():
    assert should_use_gpu({"states": {"physics": {"modality": "cassi_sci"}}}, "auto") is True

def test_auto_cpu_modality():
    assert should_use_gpu({"states": {"physics": {"modality": "ct_diagnostic"}}}, "auto") is False

def test_estimate_cost_gpu_has_keys():
    r = estimate_cost({}, True)
    assert "mode" in r and "est_cost_usd" in r

def test_estimate_cost_cpu_free():
    r = estimate_cost({}, False)
    assert r["est_cost_usd"] == "$0.00"
