import math
import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from src.utils.masking import sample_mask_batch, create_future_mask

class FC(nn.Module):
    def __init__(self, dims= [[8, 16], [16, 16], [16, 8]], dropout_rate=0.1, enable_layer_norm=True):
        super().__init__()

        n_layers = len(dims)

        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Linear_"+str(i), nn.Linear(dims[i][0], dims[i][1]))
            if enable_layer_norm and (i < n_layers - 1):
                # Do not add LayerNorm to the final Layer
                self.layers.add_module("LayerNorm_"+str(i), nn.LayerNorm(dims[i][1]))
                self.layers.add_module("ReLU_"+str(i), nn.ReLU())
            self.layers.add_module("Dropout_"+str(i), nn.Dropout(dropout_rate))

    def forward(self, x_in):    
        x_out = self.layers(x_in)
        return x_out


class TimeSeriesEncoder(nn.Module):

    def __init__(self, input_dim=10, trans_dim=64, dim_mlp=64, output_dim=8,
                 depth=4, heads=6, dropout_rate=0.4, mlp_n_layers=2, output_type=["only_tokens"]):
        super().__init__()

        self.depth = depth
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.trans_dim = trans_dim
        self.output_type = output_type
        self.emb_token = nn.Parameter(torch.randn(1, 1, self.input_dim, dtype=torch.float32))
    

        self.mlp_n_layers = mlp_n_layers
        self.dim_mlp      = dim_mlp
        self.output_dim   = output_dim

        # Project input to transformer dimension
        self.input_proj = nn.Linear(self.input_dim, self.trans_dim)

        # Transformer
        self.model = ContinuousTransformerWrapper(
            dim_in = trans_dim,
            max_seq_len = 60,
            attn_layers = Encoder(dim=self.trans_dim, depth=self.depth, heads=self.heads),
            use_abs_pos_emb = False,
            scaled_sinu_pos_emb = True,
        )

        # MLP-Head
        mlp_dims =[]
        for i in range(self.mlp_n_layers):
            if i == 0:
                mlp_dims.append([self.trans_dim, self.dim_mlp])
            elif i == self.mlp_n_layers - 1:
                mlp_dims.append([self.dim_mlp, self.output_dim])
            else:
                mlp_dims.append([self.dim_mlp, self.dim_mlp])

        self.mlp_head = FC(dims=mlp_dims, dropout_rate=self.dropout_rate)

    def forward(self, x_unpacked):
        """
        x_unpacked: (batch_size, seq_len, emb_dim)
        """
        device = x_unpacked.device
        batch_size = x_unpacked.shape[0]
        x_lengths = torch.sum(x_unpacked[..., -1] >= 0, dim=-1)

        # CLS token: (B, 1, D)
        emb_tokens = self.emb_token.expand(batch_size, -1, -1)
        x_cat = torch.cat((emb_tokens, x_unpacked), dim=1)  # (B, T+1, D)

        # Project input to match transformer dim
        x_cat = self.input_proj(x_cat)

        # Mask: True for valid positions
        mask = (torch.arange(x_cat.shape[1], device=device)[None, :] < (x_lengths + 1)[:, None])

        # Transformer forward
        z_intermediate = self.model(x_cat, mask=mask)

        if self.output_type == "only_tokens":
            # Only process the tokens, not the CLS token: BxTxD
            output = self.mlp_head(z_intermediate[:, 1:, :])  # Remove CLS token
        if self.output_type == "only_cls_tokens":
            # Just take the CLS token: BxD (D output dimension)
            output = self.mlp_head(z_intermediate[:, 0, :])
        elif self.output_type == "cls_and_tokens":
            # For each timestep T there is an token + CLS token: Bx(T+1)xD
            output = self.mlp_head(z_intermediate)
        
        return output


class MlpEncoder(nn.Module):
  
    def __init__(self, input_dim, output_dim, hidden_dim=32, n_layers=3, dropout_rate=0.1):
        super().__init__()
        
        mlp_dims =[]
        for i in range(self.n_layers):
            if i == 0:
                mlp_dims.append([input_dim, hidden_dim])
            elif i == self.n_layers - 1:
                mlp_dims.append([hidden_dim, output_dim])
            else:
                mlp_dims.append([hidden_dim, hidden_dim])

        self.net = FC(dims=mlp_dims, dropout_rate=dropout_rate, enable_layer_norm=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MaskedDecoderPredictor(nn.Module):

    def __init__(self, d_model=128, nhead=4, depth=4, dropout=0.1, max_len=50, output_type='only_tokens'):

        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.output_type = output_type

        dec_layer = nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=4*d_model, 
                dropout=dropout,
                batch_first=True, 
                norm_first=True
            )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=depth)
        # mask token (the "query" used for masked positions)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # positional embeddings for all timesteps
        self.pos_embed = ScaledSinusoidalEmbedding(d_model)
        # self.pos_table = self.pos_embed(torch.zeros(1, max_len, 1))  # (1, max_len, d_model)

    def forward(self, z, idx_vis, idx_mask, vis_key_padding_mask=None):
        """
        Inputs:
            z_ctx_full: (B, T, D)    -- full encoder output over sequence (includes masked slots set to zeros by encoder)
            idx_vis:    (B, N_vis)   -- visible token indices per sample, padded (values 0..T-1)
            idx_mask:   (B, M)       -- masked indices per sample (values 0..T-1), no padding
            vis_key_padding_mask: (B, N_vis) -- True where idx_vis is padding (mask those memory positions)
        Returns:    
            z_pred_masked: (B, M, D)
        """
        B, N, D = z.shape
        assert D == self.d_model, "unmatching feature dimension"        
        device = z.device
        # self.pos_table = self.pos_table.to(device)

        # ---------------------
        # 1) compute positional table up to needed max index
        # if type(idx_vis) is list:
        #     max_needed_idx = max( max([int(max(t)) for t in idx_vis]), int(idx_mask.max()) ) + 1
        # else:
        #     max_needed_idx = int(max(int(idx_vis.max()), int(idx_mask.max()))) + 1 
        # # TODO what is the max_needed_idx -> is this for nuScenes then 41 or can this be batch-wise ??
        # dummy_x = torch.zeros(1, max_needed_idx, 1, device=device) # just a placeholder
        # pos_table = self.pos_embed(dummy_x)  # (max_needed_idx, D)
        dummy_x = torch.zeros(1, self.max_len, 1, device=device) # just a placeholder      
        pos_table = self.pos_embed(dummy_x).to(device)

        # ---------------------
        # 2) build memory (visible embeddings gathered & their pos)
        # if False:
        #     # gather z_ctx_full at idx_vis (B, N_vis, D)
        #     idx_vis_exp = idx_vis.unsqueeze(-1).expand(-1, -1, D)  # (B, N_vis, D)
        #     z_vis_padded = torch.gather(z, 1, idx_vis_exp)  # (B, N_vis, D)
        #     # gather positional encodings for visible positions
        #     pos_vis = pos_table[idx_vis]    # (B, N_vis, D)
        #     memory = z_vis_padded + pos_vis
        # else:
        #     # The positional embeddings are already included in z from the encoder
        #     memory = z
        # ---------------------
        # 3) build target queries for masked positions (mask token + pos)
        pos_mask = pos_table[idx_mask]  # (B, M, D)
        tgt_mask = self.mask_token.expand(B, idx_mask.size(1), D) + pos_mask  # (B, M, D)

        if self.output_type == "only_tokens":
            tgt = tgt_mask
            if False:
                # Currently the setting "only_tokens" causes that the CLS token in not part of the encoder output and therefore cannot be omitted here
                # Omit CLS token again
                z = z[:,1:,:]
        elif self.output_type == "cls_and_tokens":
            # Also attend to the CLS token (idx=0)
            pos_cls = pos_table[0] 
            tgt_cls = self.mask_token.expand(B, 1, D) + pos_cls  # (B, 1, D)
            tgt = torch.cat([tgt_cls, tgt_mask], dim=1)  # (B, M+1, D)

            # vis_key_padding_mask: (B, T)    # True = PAD, False = valid
            # CLS token is always valid (never padded), so prepend a False for each row
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=vis_key_padding_mask.device)  # (B, 1)
            vis_key_padding_mask = torch.cat([cls_pad, vis_key_padding_mask], dim=1)  # (B, T+1)

        # ---------------------
        # 4) call decoder
        # memory_key_padding_mask: shape (B, N_vis) with True at padding positions
        # tgt_key_padding_mask can be None because all M queries are valid
        z_pred_masked = self.decoder(tgt=tgt, memory=z, tgt_key_padding_mask=None, memory_key_padding_mask=vis_key_padding_mask)


        return z_pred_masked


class MlpPredictor(nn.Module):
  
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout_rate=0.2, enable_layer_norm=True):
        super().__init__()

        mlp_dims =[]
        output_dim = input_dim  # Input and output dims of the predictor
        for i in range(n_layers):
            if i == 0:
                mlp_dims.append([input_dim, hidden_dim])
            elif i == n_layers - 1:
                mlp_dims.append([hidden_dim, output_dim])
            else:
                mlp_dims.append([hidden_dim, hidden_dim])

        self.MlpPredictor = FC(dims=mlp_dims, dropout_rate=dropout_rate, enable_layer_norm=enable_layer_norm)

    def forward(self, x_in):
        x_out = self.MlpPredictor(x_in)
        return x_out


class JEPA_model(nn.Module):

    def __init__(self, model_settings={}):
        super().__init__()
        self.data_source         = model_settings['data_source']
        self.apply_normalization = model_settings['apply_normalization']          
        self.input_features      = model_settings['input_features']
        self.ema_decay           = model_settings['ema_settings']['ema_decay'] 

        norm_mode_options = ['obtain_max_and_min_values', 'apply_min_max_norm']        
        self.norm_mode = norm_mode_options[1]
        if self.norm_mode == "obtain_max_and_min_values":
            self.norm_min_max_vals  = {key: [math.inf, -math.inf] for key in self.input_features}
        elif self.norm_mode == "apply_min_max_norm":
            # Min-Max-Values for all lidar objects (min. length = 8) of the total nuscenes dataset (train, val, and test)
            self.norm_min_max_vals = {'time_idx_in_scenario_frame': [0, 40],
                                      'x': [-50, 50],
                                      'y': [-50, 50],
                                      'z':  [-4.06, 7.29],
                                      'size_x': [0.23, 8.073],
                                      'size_y': [0.32, 29.882],
                                      'size_z': [0.61, 9.314],
                                      'heading': [-6.28, 6.28],
                                      'v': [0.0, 20],
                                      'pred_class': [0, 10],
                                      'tracking_score': [0.0, 1.0], 
                                      'subclass_value': [1.0, 2.0],}
        assert all([x in self.norm_min_max_vals for x in self.input_features]), "unknown feature in self.model_input_features"


        self.add_extra_feature = model_settings['add_extra_feature']
        if self.data_source == 'synthetic':
            self.input_dim = 3            
            self.input_seq_len = model_settings['input_seq_len']
        else:
            if self.add_extra_feature:
                # Add an extra feature vector to the input data (history=1, future/masked=0, padded=-1)
                self.input_dim = model_settings['input_dim'] + 1
            else:
                self.input_dim =model_settings['input_dim']

        self.latent_dim = model_settings['latent_dim']
        self.learnable_params = {}

        self.encoder_type = model_settings['encoder']['type']
        if self.encoder_type == "mlp":            
            self.mlp_n_layers = model_settings['encoder']['mlp']['n_layers']
            self.mlp_dim_mlp = model_settings['encoder']['mlp']['hidden_dim']
            self.mlp_dim_dropout_rate = model_settings['encoder']['mlp']['dropout_rate']
            self.mlp_dim_input = int(self.input_dim * (self.input_seq_len/2.0))
        elif self.encoder_type == "transformer":
            self.encoder_transformer_depth = model_settings['encoder']['transformer']['depth']
            self.encoder_transformer_heads = model_settings['encoder']['transformer']['heads']
            self.encoder_transformer_dropout = model_settings['encoder']['transformer']['dropout_rate']
            self.encoder_transformer_trans_dim = model_settings['encoder']['transformer']['trans_dim']
            self.encoder_transformer_dim_mlp = model_settings['encoder']['transformer']['dim_mlp']
            self.encoder_transformer_mlp_n_layers = model_settings['encoder']['transformer']['mlp_n_layers']
            self.encoder_transformer_output_type = model_settings['encoder']['transformer']['output_type']
        
        self.predictor_type = model_settings['predictor']['type']
        if self.predictor_type == "mlp":
            self.predictor_layers     = model_settings['predictor']['mlp']['n_layers']
            self.predictor_hidden_dim = model_settings['predictor']['mlp']['hidden_dim']
            self.predictor_dropout    = model_settings['predictor']['mlp']['dropout_rate']
            self.predictor_enable_layer_norm = model_settings['predictor']['mlp']['enable_layer_norm']
        elif self.predictor_type == "transformer_decoder":
            self.predictor_transformer_dec_depth = model_settings['predictor']['transformer_decoder']['depth']
            self.predictor_transformer_dec_heads = model_settings['predictor']['transformer_decoder']['heads']
            self.predictor_transformer_dec_dropout = model_settings['predictor']['transformer_decoder']['dropout_rate']
            self.predictor_transformer_dec_output_type = model_settings['predictor']['transformer_decoder']['output_type']
        
        self.train_masking = model_settings['train_masking']

        #########################################
        ### Encoder #############################
        if self.encoder_type == "mlp":
            self.encoder_online = MlpEncoder(input_dim=self.mlp_dim_input,
                                              hidden_dim=self.mlp_dim_mlp,
                                              output_dim=self.latent_dim,
                                              dropout_rate=self.mlp_dim_dropout_rate,
                                              n_layers=self.mlp_n_layers)
            self.encoder_target = MlpEncoder(input_dim=self.mlp_dim_input,
                                              hidden_dim=self.mlp_dim_mlp,
                                              output_dim=self.latent_dim,
                                              dropout_rate=self.mlp_dim_dropout_rate,
                                              n_layers=self.mlp_n_layers)
            
            self.learnable_params['Encoder_MLP'] = sum(p.numel() for p in self.encoder_online.parameters() if p.requires_grad)

        elif self.encoder_type == "transformer":
            # raise NotImplementedError("Transformer encoder not yet implemented")
            # Context- / Online-Encoder
            self.encoder_online = TimeSeriesEncoder(input_dim=self.input_dim, 
                                                    trans_dim=self.latent_dim, 
                                                    output_dim=self.latent_dim, 
                                                    depth=self.encoder_transformer_depth, 
                                                    heads=self.encoder_transformer_heads, 
                                                    dropout_rate=self.encoder_transformer_dropout, 
                                                    dim_mlp=self.encoder_transformer_dim_mlp, 
                                                    mlp_n_layers = self.encoder_transformer_mlp_n_layers,
                                                    output_type=self.encoder_transformer_output_type)
            # Target-Encoder
            self.encoder_target = TimeSeriesEncoder(input_dim=self.input_dim, 
                                                    trans_dim=self.encoder_transformer_trans_dim, 
                                                    output_dim=self.latent_dim, 
                                                    depth=self.encoder_transformer_depth, 
                                                    heads=self.encoder_transformer_heads, 
                                                    dropout_rate=self.encoder_transformer_dropout,
                                                    dim_mlp=self.encoder_transformer_dim_mlp, 
                                                    mlp_n_layers = self.encoder_transformer_mlp_n_layers,
                                                    output_type=self.encoder_transformer_output_type)
            self.learnable_params['Encoder_Transformer'] = sum(p.numel() for p in self.encoder_online.parameters() if p.requires_grad)
        
        # Freeze gradients for target encoder immediately
        for p in self.encoder_target.parameters():
            p.requires_grad = False
        for p in self.encoder_target.parameters():
            assert p.requires_grad == False, "Target encoder parameters must not require gradients"

        #########################################
        ### Predictor ###########################
        if self.predictor_type == "mlp":
            self.predictor = MlpPredictor(input_dim=self.latent_dim, 
                                        hidden_dim=self.predictor_hidden_dim,
                                        n_layers=self.predictor_layers,
                                        dropout_rate=self.predictor_dropout,
                                        enable_layer_norm = self.predictor_enable_layer_norm)
            self.learnable_params['Predictor_MLP'] = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        
        elif self.predictor_type == "transformer_decoder":
            
            self.predictor = MaskedDecoderPredictor(d_model=self.latent_dim,  
                                                    depth=self.predictor_transformer_dec_depth, 
                                                    nhead=self.predictor_transformer_dec_heads, 
                                                    dropout=self.predictor_transformer_dec_dropout,
                                                    output_type= self.predictor_transformer_dec_output_type)
            self.learnable_params['Predictor_Transformer_Decoder'] = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)

        self._initialize_target()


    def _initialize_target(self):
        # Copy encoder_online weights to encoder_target
        for target_param, online_param in zip(self.encoder_target.parameters(), self.encoder_online.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False  # Freeze EMA encoder


    def norm_model_input(self, x):
        B, T, F = x.shape
        
        if self.norm_mode == 'obtain_max_and_min_values':
            for b in range(B):
                len = torch.count_nonzero(x[b,:, -1])
                b_range = range(0, len)
                for i, feature in enumerate(self.input_features):
                    x_max = x[b, b_range, i].max().item()
                    x_min = x[b, b_range, i].min().item()
                    # [feature][min_val, max_val]
                    if x_max > self.norm_min_max_vals[feature][1]:
                        self.norm_min_max_vals[feature][1] = x_max
                    if x_min < self.norm_min_max_vals[feature][0]:
                        self.norm_min_max_vals[feature][0] = x_min     

        elif self.norm_mode == 'apply_min_max_norm':                
            for b in range(B):
                len = torch.count_nonzero(x[b,:, -1])
                b_range = range(0, len)

                for i, feature in enumerate(self.input_features):
                    min_val = self.norm_min_max_vals[feature][0]
                    max_val = self.norm_min_max_vals[feature][1]
                    # Apply feature specific min-max norm
                    x[b, b_range, i] = x[b, b_range, i].sub_(min_val).div_(max_val - min_val)

        else:
            assert False, "unknown norm type"
        
        return x
    

    @torch.no_grad()
    def update_target_ema(self):        
        ema_decay_applied = self.ema_decay 
        # Exponential Moving Average (EMA)
        for target_param, online_param in zip(self.encoder_target.parameters(), self.encoder_online.parameters()):
            target_param.data.mul_(ema_decay_applied).add_(online_param.data, alpha=1 - ema_decay_applied)
    
  
    def forward_train(self, x_whole):
        """
        Inputs:
            x_whole: (B, T, F)
        Returns:
            z_ctx_full:       (B, T, D)     -- Total output of context encoder
            z_ctx_full_cls:   (B, D)        -- CLS of context encoder
            z_ctx_masked:     (B, T_fut, D) -- output of predictor
            z_ctx_maksed_cls: (B, D)        -- CLS of predictor
            z_tgt_full:       (B, T, D)     -- Total output of target encoder
            z_tgt_full_cls:   (B, D)        -- CLS of target encoder
            z_tgt_masked:     (B, T, D)     -- Masked output of target encoder
            
        """
        lengths = (x_whole.abs().sum(dim=-1) != 0.0).sum(dim=1).tolist()
        B, T, F = x_whole.shape
        device = x_whole.device

        if self.apply_normalization:
            x_whole = self.norm_model_input(x_whole)

        # --- 1. sample fixed-size mask ---
        x_ctx, idx_mask, idx_vis_padded, vis_padding = sample_mask_batch(x_whole=x_whole, lengths=lengths, mask_cfg=self.train_masking, device=device, add_extra_feature=self.add_extra_feature)

        # --- 2. Context Encoder ---
        z_ctx_full = self.encoder_online(x_ctx)       # (B, T, D)  or (B, T+1, D) (if CLS token is included)
        if self.encoder_online.output_type == "only_tokens":
            z_ctx_full_cls  = "not available"
            z_ctx_full      = z_ctx_full
        elif self.encoder_online.output_type == "cls_and_tokens":
            z_ctx_full_cls  = z_ctx_full[:, 0, :]
            z_ctx_full      = z_ctx_full[:, 1:, :]

        # --- 3. Target Encoder ---
        with torch.no_grad():
            # x_whole: (B, T, F) padded input
            B, T, F = x_whole.shape
            device = x_whole.device
            # build feature mask: 1 for valid, -1 for padded (probably the -1 part is not needed as the padded inputs are masked anyway in the decoder)
            lengths_tensor = torch.as_tensor(lengths, device=device)
            if self.add_extra_feature:
                time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
                feature_mask_tgt = torch.ones((B, T), device=device)
                feature_mask_tgt[time_idx >= lengths_tensor.unsqueeze(1)] = -1.0  # padded → -1
                # concat as extra channel
                x_tgt = torch.cat([x_whole, feature_mask_tgt.unsqueeze(-1)], dim=-1)  # (B, T, F+1)
            else:
                x_tgt =x_whole

            z_tgt = self.encoder_target(x_tgt)       # (B, T, D)
            if self.encoder_target.output_type == "only_tokens":
                z_tgt_full_cls = "not available"
                z_tgt_full     = z_tgt
            elif self.encoder_target.output_type == "cls_and_tokens":
                z_tgt_full_cls = z_tgt[:, 0, :]
                z_tgt_full     = z_tgt[:, 1:, :]
            z_tgt_masked = torch.gather(z_tgt, 1, idx_mask.unsqueeze(-1).expand(-1, -1, z_tgt.size(-1)))  # (B, M, D)
            if False and self.inference_setup['token_usage'] == 'CLS':
                z_tgt_cls = z_tgt[:, 0, :].unsqueeze(1) # (B, 1, D)
                z_tgt_masked = torch.cat((z_tgt_cls, z_tgt_masked), dim=1)  # (B, M+1, D)

        # --- 4. predictor (decoder) ---
        if self.predictor.output_type == "cls_and_tokens" and self.encoder_online.output_type == "cls_and_tokens":
            z_input_predictor = torch.cat((z_ctx_full_cls.unsqueeze(dim=1), z_ctx_full), dim=1)
        else:
            z_input_predictor = z_ctx_full
        z_pred_masked = self.predictor(z_input_predictor, idx_vis_padded, idx_mask, vis_padding)  # (B, M, D)
        if self.predictor.output_type == "only_tokens":
            z_ctx_maksed_cls = "not avialble"
            z_ctx_masked     = z_pred_masked
        elif self.predictor.output_type == "cls_and_tokens":
            z_ctx_maksed_cls = z_pred_masked[:, 0, :]
            z_ctx_masked     = z_pred_masked[:, 1:, :]

        return {'z_ctx_full':       z_ctx_full,             # Total output of context encoder
                'z_ctx_full_cls':   z_ctx_full_cls,         # CLS of context encoder
                'z_ctx_masked':     z_ctx_masked,           # output of predictor
                'z_ctx_maksed_cls': z_ctx_maksed_cls,       # CLS of predictor
                'z_tgt_full':       z_tgt_full,             # Total output of target encoder
                'z_tgt_full_cls':   z_tgt_full_cls,         # CLS of target encoder
                'z_tgt_masked':     z_tgt_masked,           # Masked output of target encoder
            }
    

    def forward_pred_future(self, x_history, x_future):
        """
        Inputs:
            x_history:          (B, T_hist, F)
            x_future:           (B, T_fut, F)
        Returns:
            'z_ctx_hist':       (B, T_hist, D) -- Output of the context encoder
            'z_ctx_hist_cls':   (B, D)         -- CLS of the context encoder
            'z_ctx_future':     (B, T_fut, D)  -- Output of the predictor
            'z_ctx_future_cls': (B, D)         -- CLS of the predictor
            'z_tgt_future':     (B, T_fut, D)  -- Ouptut of the target encoder
            'z_tgt_future_cls': (B, D)         -- CLS of the target encoder
        """
        input_all_tokens_to_target = False
        B, T_hist, F = x_history.shape
        T_fut = x_future.size(1)

        
        if self.apply_normalization:
            x_history = self.norm_model_input(x_history)
            x_future  = self.norm_model_input(x_future)

        # --- 1. Prepare inputs and create masks ---
        x_history_masked, idx_future, idx_hist_list, vis_key_padding_mask, x_whole = create_future_mask(x_history, x_future, add_extra_feature=self.add_extra_feature)

        # --- 2. Context Encoder / Encode history (visible tokens) ---
        z_ctx_hist = self.encoder_online(x_history_masked)  # (B, T_hist, D)
        if self.encoder_online.output_type == "only_tokens":
            z_ctx_hist_cls = "not available"
            z_ctx_hist     = z_ctx_hist
        elif self.encoder_online.output_type == "cls_and_tokens":
            z_ctx_hist_cls = z_ctx_hist[:, 0, :]
            z_ctx_hist     = z_ctx_hist[:, 1:, :]

        # --- 3. Target Encoder / Encode future (all tokens) ----
        with torch.no_grad():
            if input_all_tokens_to_target :
                x_input_target = x_whole
            else:
                B, T, _ = x_future.shape
                if self.add_extra_feature:
                    ones = torch.ones((B, T), device=x_future.device, dtype=x_future.dtype)
                    x_input_target = torch.cat((x_future, ones.unsqueeze(dim=2)), dim=2)
                else:
                    x_input_target = x_future
            z_tgt_future = self.encoder_target(x_input_target)  # (B, T_fut, D)
            if self.encoder_target.output_type == "only_tokens":
                z_tgt_future_cls = "not available"
                z_tgt_future     = z_tgt_future
            elif self.encoder_target.output_type == "cls_and_tokens":
                z_tgt_future_cls = z_tgt_future[:, 0, :]
                z_tgt_future     = z_tgt_future[:, 1:, :]

            if input_all_tokens_to_target:
                z_future_tgt_unmasked = z_tgt_future.gather(1, idx_future.unsqueeze(-1).expand(-1, -1, z_tgt_future.size(-1)))  # (B, T_fut, D)
            else:
                z_future_tgt_unmasked = z_tgt_future
            
            
        # ---- 4. Predictor (decoder) ----
        # Treat history as "visible memory", future steps as "masked queries"
        if self.predictor.output_type == "cls_and_tokens" and self.encoder_online.output_type == "cls_and_tokens":
            z_input_predictor = torch.cat((z_ctx_hist_cls.unsqueeze(dim=1), z_ctx_hist), dim=1)
        else:
            z_input_predictor = z_ctx_hist
        z_ctx_future = self.predictor(z=z_input_predictor, idx_vis=idx_hist_list, idx_mask=idx_future, vis_key_padding_mask=vis_key_padding_mask)  # (B, T_fut, D)
        if self.predictor.output_type == "only_tokens":
            z_ctx_future_cls = "not available"
            z_ctx_future     = z_ctx_future
        elif self.predictor.output_type == "cls_and_tokens":
            z_ctx_future_cls = z_ctx_future[:, 0, :]
            z_ctx_future     = z_ctx_future[:, 1:, :]

        # --- 5. Flatten the outputs ---        
        # (B, T_fut*D) or (B, (T_fut+1)*D) if CLS token is included
        assert z_future_tgt_unmasked.shape == z_ctx_future.shape, "Shapes of target and pred do not match"      
        
        return {'z_ctx_hist':       z_ctx_hist,         # Output of the context encoder
                'z_ctx_hist_cls':   z_ctx_hist_cls,     # CLS of the context encoder
                'z_ctx_future':     z_ctx_future,       # Output of the predictor
                'z_ctx_future_cls': z_ctx_future_cls,   # CLS of the predictor
                'z_tgt_future':     z_tgt_future,       # Ouptut of the target encoder
                'z_tgt_future_cls': z_tgt_future_cls,   # CLS of the target encoder
                }   
