
import os
import wandb

from datetime import datetime

from src.dataset.data_factory import DataFactory
from src.training.training import train_jepa_framework
from src.model.model import JEPA_model
from src.loss.loss_0 import Loss_0
from src.utils.save_json_files import save_json_file
from src.evaluation.eval_0 import Evaluation_0
from src.utils.git_info import get_git_info


# ------------------------
# Configuration
# ------------------------
output_dir   = "result_folder"

data_sources  = ["synthetic_sample_data", "automotive_scene_level"]
data_source = data_sources[1]

project_name = "jepa_approach"
a = datetime.now()  
run_name = (datetime.today().strftime('%Y%m%d')  + '_' + "%s_%s" % (a.hour, a.minute) + '_' + project_name)
output_dir = os.path.join(output_dir, data_source, run_name)

general_settings = {'project_name': project_name,
                    'run_name':     run_name,
                    'output_dir':   output_dir,}

framework_tasks = ["detection_of_injected_anomalies"]
task_settings = {'framework_task':      framework_tasks[0],
                 'obj_part_for_errormodel':      'whole_object',           # ['history', 'future', 'whole_object']
                 'embeddings_4_ad':                           'encoder_max_token',      # ["encoder_max_token", "encoder_avg_token", "encoder_CLS",  "encoder_mean_var", "predictor_max_token", "predictor_avg_token", "predictor_CLS_encoder", "Tokens_future", "last_token_future", "CLS_fut_vs_pred"]
                 'error_model_params':  {'error_model':       'IndependentErrorModel',  # ['IndependentErrorModel', 'PEM']
                                         'error_injected_in': '',                       # same as ['obj_part_for_errormodel'] 
                                         'n_times_anomaly':   1,
                                         'error_statistics':  {'mean':       {'heading':0, 'v':5, },
                                                               'variance':   {'heading': 0.1, 'v': 0.1,},
                                                              },
                                         'obj_features_to_alter': ['v'],
                                        },
                 }


data_settings = {'data_source':      data_source,
                 'base_data_path':   os.path.join('data', 'nuscenes'),
                 'output_dir':       output_dir,
                 'synthetic_settings': {'seq_len':          30,
                                        'input_dim':        3,
                                        'seq_len_hist':     25,
                                        'seq_len_pred':     5,
                                        'n_times_anomaly':  1,
                                        'anomaly_noise':    {'mu': 0.5, 'sigma': 0.2},
                                        'anomaly_ratio':    0.5,
                                        'save_one_sample':  True,
                                        'output_dir':       output_dir,}, 

                 'scene_level_settings': {'min_obj_length':       8,  
                                          'seq_len_pred':         4,
                                          'useEgoCentricCoord':   True,
                                          'obj_features_to_load': [],       # empty list = all features
                                          'sensor_modality':      'lidar',  # 'lidar' or 'camera'
                                          # Available features: ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'pred_class', 'v', 'v_x', 'v_y', 'tracking_score']
                                          'input_features_for_model': ['x', 'y', 'v', 'heading'],
                                          'output_dir':           output_dir,
                                        },
   
                }

    
encoder_types   = ["mlp", "transformer"]
predictor_types = ["mlp", "transformer_decoder"]
latent_dim = 32
model_settings =    {'data_source':         data_source,
                     'latent_dim':          latent_dim,
                     'input_seq_len':       50,
                     'batch_size':          1,      # The batch size is varying as one scene contains a variable number of objects 
                     'apply_normalization': False,
                     'input_features':      data_settings['scene_level_settings']['input_features_for_model'],
                     'input_dim':           len(data_settings['scene_level_settings']['input_features_for_model']),
                     'CLS_4_AD':            False,
                     'add_extra_feature':   True,   # Feature Vector [BxTx1] added to input data (1=Available, 0=Masked, -1=Padded)
                     
                     'encoder': {'type':  encoder_types[1],
                                 'mlp':  {
                                     'n_layers':     3,
                                     'hidden_dim':   32,
                                     'dropout_rate': 0.1,
                                     # output_dim = 'latent_dim' 
                                        },
                                 'transformer':  {
                                        'depth':        5,
                                        'heads':        10,
                                        'trans_dim':    latent_dim,   # must be the same as trans_dim for predictor due to the sinousoidal pos emb
                                        'output_type':  "cls_and_tokens",  #["only_tokens", "only_cls_tokens", "cls_and_tokens"]
                                        # Input and output dims of mlp are 'latent_dim'
                                        'mlp_n_layers': 3,
                                        'dropout_rate': 0.2,
                                        'dim_mlp':      32,
                                                },
                                  },
                     'predictor': {'type': predictor_types[1],
                                   'mlp': {
                                       'n_layers':          4,
                                       'hidden_dim':        64,
                                       'dropout_rate':      0.2,
                                       'enable_layer_norm': True,
                                            },
                                    'transformer_decoder':  {
                                        'depth':        3,
                                        'heads':        4,
                                        'dropout_rate': 0.1,
                                        'output_type':  "cls_and_tokens",  #["only_tokens", "only_cls_tokens", "cls_and_tokens"]
                                        },
                                    'transformer_encoder':  {
                                        'depth':                 1,
                                        'heads':                 2,
                                        'trans_dim':             32,
                                        'only_output_cls_token': False,
                                        # Input and output dims of mlp are 'latent_dim'
                                        'mlp_n_layers': 2,
                                        'dropout_rate': 0.2,
                                        'dim_mlp':      32,
                                        },
                                  },
                     'train_masking':    {'type':           'fixed_masking',   # 'fixed_masking' or 'ratio_masking'
                                          'masking_ratio':  0.4,
                                          'fixed_num_mask': 4,},
                     'ema_settings':    {},
                    }

training_settings = {'data_source':         data_source,
                     'use_amp':             False,
                     'epochs':              800,
                     'optimizer_type':      'Adam',       # ['Adam', 'AdamW']
                     'learning_rate':       3e-5, 
                     'ema_decay':           0.995,
                     'output_dir':          output_dir,}
model_settings['ema_settings']['ema_decay'] = training_settings['ema_decay']
# Training settings:
# Full train-set (700 scenes):      lr=3e-5     ema_decay=0.999
# Reduced train-set (100 scenes):   lr=1e-4     ema_decay=0.995
training_settings['epochs_to_evaluate'] = [i for i in range((training_settings['epochs'])) if i % 25 == 0]

inputs_variations = ['diff', 'concat_diff', 'concat_all', 'concat_z1_z2']
evaluation_settings = {'output_dir':  output_dir,
                       'data_source': data_source,    
                       'CLS_4_AD':    model_settings['CLS_4_AD'],
                       'eval_input':  inputs_variations[0],
                       'latent_dim':  model_settings['latent_dim'],
                       'anomaly_detection': 
                                {'lof':  {'enabled':     True,
                                          'n_neighbors': 15,
                                          'novelty':     True},
                                 'abod': {'enabled': True, 
                                          'method': 'unify'},
                                 'gmm':  {'enabled':         True,
                                          'n_components':    5,
                                          'covariance_type': 'full',
                                          'log_likelihood_threshold_percentil': 5,},
                                },
                       'perform_anomaly_detection': True,
                       'perform_dim_reduction':     True, 
                       'save_latent_embeddings':    True,
                       'save_model':                True,
                      }

loss_types = ["L2_loss", "L1_loss"]
loss_settings = {'type': loss_types[1]}

git_metadata = get_git_info()
config  =  {'run_name':     run_name,
            'general':      general_settings,
            'task':         task_settings,
            'data':         data_settings,
            'model':        model_settings,
            'training':     training_settings,
            'evaluation':   evaluation_settings,
            'loss':         loss_settings,
            'git_info':     git_metadata,
}

# ------------------------
# Dataset
# ------------------------
data_factory = DataFactory(config)
dataloader_train = data_factory.get_dataloader(data_split='train')
dataloader_test  = data_factory.get_dataloader(data_split='test')


# ------------------------
# Model & Optimizer
# ------------------------
model = JEPA_model(model_settings = config['model'],)
config['model']['learnable_params'] = model.learnable_params


# ------------------------
# Evaluator
# ------------------------
evaluator_0 = Evaluation_0(dataloader_train     = dataloader_train,
                           dataloader_test      = dataloader_test,
                           evaluation_settings  = evaluation_settings,
                           task_settings        = task_settings)

loss = Loss_0(loss_settings)

# Initialize wandb
run = wandb.init(
        project = project_name,
        notes   = "",
        tags    = ["baseline", "paper1"],
        config  = config
    )
wandb_settings = {'project_name':   project_name,
                  'run_name_orig':  run.name,
                  'run_id':         run.id,
                  'link':           'https://wandb.ai/' + run.path}




      
### Save settings for this run
save_json_file(output_dir=output_dir, filename="settings.json", data=config) 
    
# ------------------------
# Start a new training
# ------------------------
train_jepa_framework(model, dataloader_train, evaluator_0, loss, config['training'])


print("[INFO] Training and evaluation completed.")

