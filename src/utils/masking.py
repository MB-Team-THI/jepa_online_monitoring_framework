import torch
from typing import Dict, Tuple

def sample_mask_batch(x_whole, lengths, mask_cfg, device, add_extra_feature=False):
    """
    Inputs:
        x_whole: (B, T, F) padded input (zeros on padded timesteps)
        lengths: list or LongTensor (B,) lengths of each sequence in the batch
        mask_cfg: dict, configuration for masking
        device
    
    Returns:
        x_ctx: (B, T, F+1) masked input for context encoder (with mask feature)
        idx_mask: LongTensor (B, M) -> indices (0..T-1) of masked timesteps for each sample
        idx_vis:  LongTensor (B, N_vis_max) -> indices (0..T-1) of visible timesteps for each sample, padded with zeros
        vis_key_padding_mask: BoolTensor (B, T) -> True where idx_vis is   
    """
    device = device or x_whole.device
    B, T, F = x_whole.shape
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    min_len = int(lengths.min().item())
    
    # TDOO okay now M is depending on the min_len, but should we calc M per sample instead?
    if mask_cfg["type"] == "ratio_masking":
        M = max(1, int(round(min_len * mask_cfg["masking_ratio"])))
    elif mask_cfg["type"] == "fixed_masking":
        M = min(mask_cfg["fixed_num_mask"], min_len - 1)
    else:
        raise ValueError("Unknown mask_cfg['type']")

    # for each sample produce a random permutation among valid indices [0..L-1]
    idx_mask_list = []
    idx_vis_list = []
    n_vis_list = []
    for b in range(B):
        L = int(lengths[b].item())
        valid_idx = torch.arange(L, device=device)        
        if True:
            # Each mask is random idepently
            perm = valid_idx[torch.randperm(L, device=device)]
            idx_mask_b = perm[:M]            # (M,)
            idx_vis_b = perm[M:]             # (L - M,)
        else:
            # A mask block of length M
            masking_start_idx = perm = valid_idx[torch.randperm(L-M, device=device)][0].item()
            idx_mask_b = valid_idx[masking_start_idx:masking_start_idx+4]
            idx_vis_b  = torch.cat((valid_idx[:masking_start_idx],valid_idx[masking_start_idx+M:]))
        idx_mask_list.append(idx_mask_b)
        idx_vis_list.append(idx_vis_b)
        n_vis_list.append(idx_vis_b.size(0))

    # pad idx_vis to the same N_vis_max across batch
    N_vis_max = max(n_vis_list)
    # idx_mask: (B, M) stack straightforward
    batch_idx_mask = torch.stack([t.clone() for t in idx_mask_list], dim=0)  # (B, M)
    
    # idx_vis padded into (B, N_vis_max)
    idx_vis_padded = torch.zeros((B, N_vis_max), dtype=torch.long, device=device)
    for b in range(B):
        n = idx_vis_list[b].size(0)
        idx_vis_padded[b, :n] = idx_vis_list[b]

    # idx_vis padded into (B, T)
    vis_key_padding_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    for b in range(B):
        n = idx_vis_list[b].size(0) + M
        assert n == lengths[b], "length of visible tokens (including the masked ones / + M) must be the same as the object length"
        vis_key_padding_mask[b, :n] = False # not padding

    # --- Build x_ctx: mask those timesteps in the input (only valid timesteps)
    x_ctx = x_whole.clone()
    for b in range(B):
        mask_idx = batch_idx_mask[b]                             # (M,)
        x_ctx[b, mask_idx, :] = 0.0                        # replace masked timesteps by zeros (or learnable token)

    if add_extra_feature:
        # Optionally add a binary mask channel to x_ctx as extra feature
        # feature_mask_ctx: 1 = visible, 0 = masked, -1 = padded
        feature_mask_ctx = torch.ones((B, T), device=device, dtype=x_whole.dtype)
        # mark masked
        for b in range(B):
            feature_mask_ctx[b, batch_idx_mask[b]] = 0.0
        # mark padded timesteps
        padded_mask = (torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1))
        feature_mask_ctx[padded_mask] = -1.0
        # concat as extra feature channel
        x_ctx = torch.cat([x_ctx, feature_mask_ctx.unsqueeze(-1)], dim=2)  # (B, T, F+1)

    return x_ctx, batch_idx_mask, idx_vis_padded, vis_key_padding_mask



def create_future_mask(x_history, x_future, add_extra_feature=False):
    '''
    Inputs:
        x_history: (B, T_hist, F) padded input (zeros on padded timesteps)
        x_future:  (B, T_fut, F) future part to predict
    Returns:
        x_history_masked: (B, T_hist + T_fut, F+1) masked input for context encoder (with mask feature)
        idx_future:      LongTensor (B, T_fut)   -> indices (0..T_hist + T_fut -1) of future timesteps for each sample
        vis_key_padding_mask: BoolTensor (B, T_hist + T_fut) -> True where idx_vis is padding (to be masked in memory)
        x_whole:         (B, T_hist + T_fut, F+1) unmasked input for target encoder (with mask feature) 
    Notes:
        x_history has variable length and is padded with zeros which makes the indexing more complex
    '''
    
    length_history = lengths = (x_history.abs().sum(dim=-1) != 0.0).sum(dim=1)
    B, T_hist, F = x_history.shape
    _, T_fut, _ = x_future.shape
    device = x_history.device

    # --- Build x_whole: concat history + future + add mask feature
    x_whole_list = []
    idx_hist_list = []
    idx_future_list = []
    vis_key_padding_mask = torch.ones((B, T_hist + T_fut), dtype=torch.bool, device=device)
    for b in range(B):
        comb = torch.cat((x_history[b, :length_history[b], :], x_future[b], x_history[b, length_history[b]:, :]), dim=0)

        if add_extra_feature:
            # Add extra feature (1 for hist and -1 for timesteps to predict)
            add_features = torch.cat((torch.ones(len(x_history[b, :length_history[b], :]), device=device),
                                    torch.ones(len(x_future[b]), device=device),
                                    -1*torch.ones(len(x_history[b, length_history[b]:, :]), device=device)), dim=0)
            comb = torch.cat((comb, add_features.unsqueeze(dim=1)), dim=1)

        vis_key_padding_mask[b, :length_history[b]+T_fut] = False
        idx_hist = torch.arange(length_history[b], device=device)
        idx_fut  = length_history[b] + torch.arange(T_fut, device=device)

        idx_hist_list.append(idx_hist.tolist())
        idx_future_list.append(idx_fut)
        x_whole_list.append(comb)        
        
    x_whole     = torch.stack(x_whole_list, dim=0)
    idx_future  = torch.stack(idx_future_list, dim=0)  # (B, T_fut)


    # --- Build x_history_masked: mask out future part in the history (input for the context encoder)
    x_history_masked = x_whole.clone()
    for b in range(B):
        # mask out future part in the history 
        x_history_masked[b, idx_future[b], :] = 0.0

    return x_history_masked, idx_future, idx_hist_list, vis_key_padding_mask, x_whole