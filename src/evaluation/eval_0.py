import os
import wandb
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

from src.utils.visualizations import visualize_latent_space
from src.utils.save_json_files import save_json_file

from src.utils.anomaly_detection_methods import perform_anomaly_detection


def copy_metrics_to_wandb(epoch, metrics_dict, prefix=""):
    log_dict = {'Epoch':            epoch,
                'LOF_accuracy':     metrics_dict['lof']['metrics_test']['accuracy_overall'],
                'LOF_F1_score':     metrics_dict['lof']['metrics_test']['f1_score'],
                'LOF_ROCAUC':       metrics_dict['lof']['metrics_test']['roc_auc_score'],
                'ABOD_accuracy':    metrics_dict['abod']['metrics_test']['accuracy_overall'],
                'ABOD_F1_score':    metrics_dict['abod']['metrics_test']['f1_score'],
                'ABOD_ROCAUC':      metrics_dict['abod']['metrics_test']['roc_auc_score'],
                'GMM_accuracy':     metrics_dict['gmm']['metrics_test']['accuracy_overall'],
                'GMM_F1_score':     metrics_dict['gmm']['metrics_test']['f1_score'],
                'GMM_ROCAUC':       metrics_dict['gmm']['metrics_test']['roc_auc_score'],
            }
    
    wandb.log(log_dict)




class Evaluation_0():
    def __init__(self, dataloader_train, dataloader_test, evaluation_settings, task_settings):
        self.dataloader_train = dataloader_train
        self.dataloader_test  = dataloader_test

        
        # Evaluation Setting
        self.output_dir                 = evaluation_settings['output_dir']
        self.data_source                = evaluation_settings['data_source']
        self.input_variation            = evaluation_settings['eval_input']
        self.CLS_4_AD                   = evaluation_settings['CLS_4_AD']
        self.anomaly_detection_settings = evaluation_settings['anomaly_detection']
        self.latent_dim                 = evaluation_settings['latent_dim']

        self.perform_anomaly_detection = evaluation_settings['perform_anomaly_detection']
        self.perform_dim_reduction     = evaluation_settings['perform_dim_reduction']
        self.save_latent_embeddings    = evaluation_settings['save_latent_embeddings']
        self.save_model                = evaluation_settings['save_model']

        # Task Settings
        self.framework_task    = task_settings['framework_task']
        self.error_injected_in = task_settings['error_model_params']['error_injected_in']
        self.embeddings_4_ad   = task_settings['embeddings_4_ad']
        if self.embeddings_4_ad in ["encoder_max_token", "encoder_avg_token", "encoder_CLS", "predictor_max_token", "predictor_avg_token", "predictor_CLS"]:
            # Use only the embeddings from the context branch
            self.res_from_both_branches = False
        elif self.embeddings_4_ad in ["Tokens_future", "last_token_future", "CLS_fut_vs_pred"]:
            # Use embeddings from the context and target branch
            self.res_from_both_branches = True
        else:           
            raise ValueError(f"Unkown parameter: self.embeddings_4_ad  = {self.embeddings_4_ad}")
        
        assert self.framework_task in ['detection_of_injected_anomalies'], "unkown framework task"
        assert self.error_injected_in in ['future','history', 'whole_object'], "unkown error_injected_in"

        self.model_info = {}
 

    def load_datasplit(self, model, dataloader, device):
        '''
        - Load data from dataloader   
            - train dataloader only provides normal data (without anomalies)
            - test/val dataloader provide normal + anomaly data
        - Execute model -> obtain embeddings and labels
        - Return results
        '''
        model.eval()

        ### Gather data
        with torch.no_grad():
            
            ### Obtain train data - without anomalies
            list_z_future, list_z_future_pred, list_z_context_special, list_labels = [], [], [], []
            for idx, batch in enumerate(dataloader):
                labels   = batch['objects']['label']
                if len(labels) == 0:
                    # No objects remain in this batch from the preprocessing step
                    continue
                x_hist   = batch['objects']['past'].to(device)
                x_future = batch['objects']['future'].to(device)
                x_whole  = batch['objects']['whole'].to(device)

                if model.data_source == 'automotive_scene_level':
                    # Scene-based dimensions: BatchSize(=1) x Samples x TimeSteps x Features --> [BatchSize x TimeSteps x Features]
                    if len(x_hist.shape) == 4:
                        # Remove the sequence dimension if it exists
                        x_hist   = x_hist.squeeze()
                        x_future = x_future.squeeze()
                        x_whole  = x_whole.squeeze()
                        labels   = labels.squeeze()
                    if len(x_hist.shape) < 3:
                        # Add additional dimension
                        x_hist   = x_hist.unsqueeze(dim=0)  
                        x_future = x_future.unsqueeze(dim=0)
                        x_whole  = x_whole.unsqueeze(dim=0)
                        labels   = labels.unsqueeze(dim=0)
                assert len(x_hist.shape) == 3, "Dimension must be [B, T, F]"

                list_labels.extend(labels.tolist())

                ### Fault injection in the Future
                if self.framework_task in ["detection_of_injected_anomalies"]:     

                    if self.error_injected_in == "future":
                        # Fault injection in the future: z_future (with anomaly) and z_future_pred (normal) 
                        z_dict = model.forward_pred_future(x_history=x_hist, x_future=x_future)
                        B, T_fut, D = z_dict['z_ctx_future'].shape

                        if self.res_from_both_branches:
                            #== Compare the results of context and target branch                             
                            if self.embeddings_4_ad == "Tokens_future":
                                list_z_future.extend(z_dict['z_tgt_future'].view(B, -1).tolist())       # contains the anomaly
                                list_z_future_pred.extend(z_dict['z_ctx_future'].view(B, -1).tolist())  # based on normal data
                            elif self.embeddings_4_ad == "last_token_future":
                                list_z_future.extend(z_dict['z_tgt_future'][:, -1, :].tolist())         # based on normal data
                                list_z_future_pred.extend(z_dict['z_ctx_future'][:, -1,:].tolist())     # contains the anomaly
                            elif self.embeddings_4_ad == "CLS_fut_vs_pred":
                                z_future_cls      = z_dict['z_tgt_future_cls']                          # CLS token / based on normal data / from the target encoder
                                z_future_pred_cls = z_dict['z_ctx_future_cls']                          # CLS token / contains the anomaly / from the context predictor
                                list_z_future.extend(z_future_cls.tolist())
                                list_z_future_pred.extend(z_future_pred_cls.tolist())  
                        else:
                            # I need some result from the target branch here to introduce the error
                            raise ValueError(f"self.error_injected_in == 'future' does not work with the ['encoder_max_token', 'encoder_avg_token', 'encoder_CLS', 'predictor_max_token', 'predictor_avg_token', 'predictor_CLS'] as self.embeddings_4_ad")
      
                    elif self.error_injected_in == "history":
                        # Fault injection in the history: z_future (normal) and z_future_pred (with anomaly) 
                        z_dict = model.forward_pred_future(x_history=x_hist, x_future=x_future)                 
                        B, T_fut, D = z_dict['z_ctx_future'].shape

                        if self.res_from_both_branches:
                            #== Compare the results of context and target branch 
                            if self.embeddings_4_ad == "Tokens_future":
                                list_z_future.extend(z_dict['z_tgt_future'].view(B, -1).tolist())       # Target embeddings  - based on normal data
                                list_z_future_pred.extend(z_dict['z_ctx_future'].view(B, -1).tolist())  # Context embeddings - contains the anomaly
                            elif self.embeddings_4_ad == "last_token":
                                list_z_future.extend(z_dict['z_tgt_future'][:, -1,:].tolist())          # Target embeddings  - based on normal data
                                list_z_future_pred.extend(z_dict['z_ctx_future'][:, -1,:].tolist())     # Context embeddings - contains the anomaly
                            elif self.embeddings_4_ad == "CLS_fut_vs_pred":
                                z_future_cls      = z_dict['z_tgt_future_cls']                          # CLS token / based on normal data / from the target encoder
                                z_future_pred_cls = z_dict['z_ctx_future_cls']                          # CLS token / contains the anomaly / from the context predictor
                                list_z_future.extend(z_future_cls.tolist())
                                list_z_future_pred.extend(z_future_pred_cls.tolist())  
                        else:
                            #== Only use the results from the context branch                            
                            # Embeddings from the context encoder
                            if self.embeddings_4_ad == "encoder_max_token":
                                z_max = torch.max(z_dict['z_ctx_hist'], dim=1).values.cpu().numpy()     # obtain max values
                                list_z_context_special.extend(z_max)
                            elif self.embeddings_4_ad == "encoder_avg_token":
                                z_avg = torch.mean(z_dict['z_ctx_hist'], dim=1).cpu().numpy()           # Average over tokens           
                                list_z_context_special.extend(z_avg.tolist())
                            elif self.embeddings_4_ad == "encoder_CLS":
                                z_cls = z_dict['z_ctx_hist_cls']                                        # CLS token is at index 0
                                list_z_context_special.extend(z_cls.tolist())
                            elif False:
                                # Flatten all tokens and apply zero-padding ?  -> same for whole_object
                                pass
                            # Embeddings from the context predictor
                            elif self.embeddings_4_ad == "predictor_max_token":
                                z_max = torch.max(z_dict['z_ctx_future'], dim=1).values.cpu().numpy()   # obtain max values
                                list_z_context_special.extend(z_max)
                            elif self.embeddings_4_ad == "predictor_avg_token":
                                z_avg = torch.mean(z_dict['z_ctx_future'], dim=1).cpu().numpy()         # Average over tokens           
                                list_z_context_special.extend(z_avg.tolist())
                            elif self.embeddings_4_ad == "predictor_CLS":
                                z_cls = z_dict['z_ctx_future_cls']                                      # CLS token is at index 0
                                list_z_context_special.extend(z_cls.tolist())
                            elif False:
                                # Flatten all tokens and apply zero-padding ?  -> same for whole_object
                                pass

                    elif self.error_injected_in == "whole_object":
                        # Fault injection somewhere in the whole object
                        z_dict = model.forward_pred_future(x_history=x_whole, x_future=x_future)   
                        # z_dict['z_ctx_hist'] is equivilant to z_ctx_whole as x_whole is provided as input
                        B, T_fut, D = z_dict['z_ctx_hist'].shape
                        
                        if not self.res_from_both_branches:
                            #== Only use the results from the context branch                         
                            # Embeddings from the context encoder
                            if self.embeddings_4_ad == "encoder_max_token":
                                z_max = torch.max(z_dict['z_ctx_hist'], dim=1).values.cpu().numpy()     # obtain max values
                                list_z_context_special.extend(z_max)
                            elif self.embeddings_4_ad == "encoder_avg_token":
                                z_avg = torch.mean(z_dict['z_ctx_hist'], dim=1).cpu().numpy()           # Average over tokens           
                                list_z_context_special.extend(z_avg.tolist())
                            elif self.embeddings_4_ad == "encoder_CLS":
                                z_cls = z_dict['z_ctx_hist_cls']                                        # CLS token is at index 0
                                list_z_context_special.extend(z_cls.tolist())
                            elif self.embeddings_4_ad == "encoder_mean_var":
                                z_mean     = torch.mean(z_dict['z_ctx_hist'], dim=1)
                                z_var      = torch.var(z_dict['z_ctx_hist'], dim=1)
                                z_mean_var = torch.cat((z_mean, z_var),dim=1).cpu().numpy()
                                list_z_context_special.extend(z_mean_var)
                            # Embeddings from the context predictor
                            elif self.embeddings_4_ad == "predictor_max_token":
                                z_max = torch.max(z_dict['z_ctx_future'], dim=1).values.cpu().numpy()   # obtain max values
                                list_z_context_special.extend(z_max)
                            elif self.embeddings_4_ad == "predictor_avg_token":
                                z_avg = torch.mean(z_dict['z_ctx_future'], dim=1)                       # Average over tokens
                                list_z_context_special.extend(z_avg.tolist())
                            elif self.embeddings_4_ad == "predictor_CLS":
                                z_cls = z_dict['z_ctx_future_cls']                                      # CLS token is at index 0
                                list_z_context_special.extend(z_cls.tolist())

                        else:
                            raise ValueError(f"self.error_injected_in == 'whole_object' is not available for the settings ['Tokens_future', 'last_token_future', 'CLS_fut_vs_pred'] as self.embeddings_4_ad")



                else:
                    raise ValueError(f"self.framework_task")

                
                if self.res_from_both_branches:
                    assert len(list_labels) == len(list_z_future) == len(list_z_future_pred), "Must be of same length"
                else:
                    assert len(list_labels) == len(list_z_context_special), "Must be of same length"


                if self.model_info == {} and False:
                    # Obtain number of FLOPS required for inference
                    batch_num_flops = FlopCountAnalysis(model=model, inputs=(x_hist, x_future)).total()
                    self.model_info['flops_per_object'] = int(batch_num_flops / x_hist.shape[0])
                    self.model_info['num_params_total_by_fvcore'] = parameter_count(model)[""]
                    self.model_info['num_params_total_by_model.parameters()'] = sum(p.numel() for p in model.parameters())
                    self.model_info['num_trainable_params_total_by_model.parameters()'] = sum(p.numel() for p in model.parameters() if p.requires_grad)                
                    save_json_file(output_dir=output_dir_epoch, filename="model_params.json", data=self.model_info) 
            
        return list_z_future, list_z_future_pred, list_z_context_special, list_labels


    def evaluate(self, model, epoch=0, training_phase="", post_eval_output_dir=""):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if post_eval_output_dir == "":
            output_dir = self.output_dir
        else:
            output_dir = post_eval_output_dir

        if training_phase == "":
            output_dir_epoch = os.path.join(output_dir, 'epoch_'+str(epoch))
        else:
            output_dir_epoch = os.path.join(output_dir, training_phase+'_epoch_'+str(epoch))

        train_z_future, train_z_future_pred, train_z, _ = self.load_datasplit(model=model, dataloader=self.dataloader_train, device=device)
        test_z_future, test_z_future_pred, test_z, test_label = self.load_datasplit(model=model, dataloader=self.dataloader_test, device=device)


        if self.res_from_both_branches:
            # Use z_future and z_future_pred for AD
            train_z_future_np      = np.array(train_z_future)
            train_z_future_pred_np = np.array(train_z_future_pred)
            train_z_diff           = train_z_future_np - train_z_future_pred_np

            test_z_future_np      = np.array(test_z_future)
            test_z_future_pred_np = np.array(test_z_future_pred)        
            test_z_diff           = test_z_future_np - test_z_future_pred_np

            if self.input_variation == "diff":
                # Only z_diff
                train_z = train_z_diff
                test_z  = test_z_diff
            elif self.input_variation == "concat_diff":
                # Concat: (z_fut, z_diff)
                train_z = np.concatenate((train_z_future_np, train_z_diff),axis=1)
                test_z  = np.concatenate((test_z_future_np, test_z_diff),axis=1)
            elif self.input_variation == "concat_all":
                # Concat: (z_fut, z_fut_pred, z_diff)
                train_z = np.concatenate((train_z_future_np, train_z_future_pred_np, train_z_diff),axis=1)
                test_z  = np.concatenate((test_z_future_np, test_z_future_pred_np, test_z_diff),axis=1)
            elif self.input_variation == "concat_z1_z2":
                # Concat: (z_fut, z_fut_pred)
                train_z = np.concatenate((train_z_future_np, train_z_future_pred_np),axis=1)
                test_z  = np.concatenate((test_z_future_np, test_z_future_pred_np),axis=1)
            else:
                assert False, "undefined self.input_variation: " + self.input_variation
        
        else:
            #== Only use the results from the context branch
            train_z = np.array(train_z)
            test_z = np.array(test_z)
        

        ### Perform visualizations
        if self.perform_dim_reduction:
            visualize_latent_space(z=test_z, labels=test_label, output_dir=output_dir_epoch)

        res_all = {}
        ### Perform Anomaly Detection and Calculate Metrics
        if self.perform_anomaly_detection:
            self.anomaly_detection_settings['output_dir'] = output_dir_epoch
            res_all = perform_anomaly_detection(x_train        = train_z, 
                                                x_test         = test_z, 
                                                y_test         = test_label, 
                                                settings       = self.anomaly_detection_settings,
                                                framework_task = self.framework_task)
            

        ### Save results 
        
        # Save resutls as json-file
        save_json_file(output_dir=output_dir_epoch, filename="results.json", data=res_all)


        # Save latent embeddings
        if self.save_latent_embeddings:
            # Convert all to python-list
            data_out = {'train': {'train_z_future':         train_z_future,
                                  'train_z_future_pred':    train_z_future_pred,
                                  'train_z':                train_z.tolist()},
                        'test':  {'test_z_future':          test_z_future,
                                  'test_z_future_pred':     test_z_future_pred,
                                  'test_z':                 test_z.tolist()},
                        'settings': {'input_variation': self.input_variation,}
                        }
            save_json_file(output_dir=output_dir_epoch, filename="latent_embeddings.json", data=data_out)

        # Save model
        if self.save_model:
            model_name = "model_epoch_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), os.path.join(output_dir_epoch, model_name))

        # Track metrics in wandb
        copy_metrics_to_wandb(epoch=epoch, metrics_dict=res_all)


        model.train()