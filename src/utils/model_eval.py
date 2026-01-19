import torch

def has_nan_inf(t):
    return torch.isnan(t).any().item(), torch.isinf(t).any().item()

def print_nan_info(name, t):
    n, inf = has_nan_inf(t)
    if n or inf:
        print(f"*** {name} contains NaN/Inf -> NaN={n}, Inf={inf}. min/max/mean:",
              float(t.min()), float(t.max()), float(t.mean()))
    else:
        print(f"{name} is OK. min/max/mean:",
              float(t.min()), float(t.max()), float(t.mean()))
