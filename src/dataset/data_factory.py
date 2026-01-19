
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .toy_dataset import get_synthetic_dataset, JEPA_TimeSeriesDataset

from src.utils.data_processing import filter_object_features
from src.dataset.loaders._load_scene_level_data import _load_scene_level_data
from src.utils.error_models import PEM, IndependentErrorModel


class DataFactory:
    def __init__(self, config):
        self.config = config
        self.batch_size = config['model']['batch_size']
        self.data_source = config['data']['data_source']

        if self.data_source == 'synthetic':
            self.data_input_dim = config['data']['synthetic_settings']['input_dim']
            self.data_seq_len   = config['data']['synthetic_settings']['seq_len']
            self.seq_len_hist   = config['data']['synthetic_settings']['seq_len_hist']
            self.seq_len_pred   = config['data']['synthetic_settings']['seq_len_pred']
        
        elif self.data_source == 'automotive_scene_level':
            self.data_input_dim     = len(config['data']['scene_level_settings']['input_features_for_model'])
            self.data_seq_len       = 50 # Placeholder
            # self.scene_level_settings = config['data']['scene_level_settings']
            self.min_obj_length     = config['data']['scene_level_settings']['min_obj_length']
            self.seq_len_pred       = config['data']['scene_level_settings']['seq_len_pred']
            self.useEgoCentricCoord = config['data']['scene_level_settings']['useEgoCentricCoord']
            self.sensor_modality    = config['data']['scene_level_settings']['sensor_modality']

            # self.pem_parameters = config['data']['scene_level_settings']['pem_parameters']
            # self.pem_obj_features_to_alter = config['data']['scene_level_settings']['pem_obj_features_to_alter']
            # self.pem_n_times_anomaly = config['data']['scene_level_settings']['pem_n_times_anomaly']
            
            self.input_features_for_model = config['data']['scene_level_settings']['input_features_for_model']

        elif self.data_source == 'automotive_object_level':
            pass

        ### Framework Task
        self.framework_task         = config['task']['framework_task']
        self.error_model_parameters = config['task']['error_model_params']
        self.error_model_parameters['error_injected_in'] = config['task']['obj_part_for_errormodel']



    def get_dataloader(self, data_split):

        if self.data_source == 'synthetic':
            x, y = get_synthetic_dataset(with_anomalies=(data_split in ['test', 'val']), config=self.config['data']['synthetic_settings'])
            dataset = JEPA_TimeSeriesDataset(x, y, self.config['data']['synthetic_settings'])
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(data_split=='train'))


        elif self.data_source == 'automotive_scene_level':
            ### Dataloader for data on scene level (all objects from camera, lidar and GT within the scene are present within one sample)
            path = os.path.join(self.config['data']['base_data_path'], 'scene_level')
            if data_split == 'train':
                path = os.path.join(path, 'train', 'base')
                data_type = 'normal'
            elif data_split == 'val':
                path = os.path.join(path, 'val', 'base')
                data_type = 'normal_and_anomalies'
            elif data_split == 'test':
                path = os.path.join(path, 'test', 'base')
                data_type = 'normal_and_anomalies'
            dataset = MatFolderDataset(folder_path=path, 
                                       data_source              = self.data_source,
                                       sensor_modality          = self.sensor_modality,
                                       seq_len_pred             = self.seq_len_pred,
                                       min_obj_length           = self.min_obj_length,
                                       useEgoCentricCoord       = self.useEgoCentricCoord,
                                       input_features_for_model = self.input_features_for_model,
                                       data_type                = data_type,
                                       framework_task           = self.framework_task,
                                       error_model_parameters   = self.error_model_parameters)
            # dataloader = dataloader_scene_level(dataset=dataset, batch_size=self.batch_size,shuffle=(data_split=='train'))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(data_split=='train'))

        elif self.data_source == 'automotive_object_level':
            ### Dataloader for data on object level (one sample is a object from camera, lidar or GT)
            path = os.path.join(self.config['data']['base_data_path'], 'object_level')
            if data_split == 'train':
                path = os.path.join(path, 'train', 'base')
                data_type = 'normal'
            elif data_split == 'val':
                path = os.path.join(path, 'val', 'base')
                data_type = 'normal_and_anomalies'
            elif data_split == 'test':
                path = os.path.join(path, 'test', 'base')
                data_type = 'normal_and_anomalies'
            dataset = MatFolderDataset(folder_path=path, 
                                       data_source              = self.data_source,
                                       sensor_modality          = self.sensor_modality,
                                       seq_len_pred             = self.seq_len_pred,
                                       min_obj_length           = self.min_obj_length,
                                       useEgoCentricCoord       = self.useEgoCentricCoord,
                                       input_features_for_model = self.input_features_for_model,
                                       data_type                = data_type,
                                       framework_task           = self.framework_task,
                                       error_model_parameters   = self.error_model_parameters)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(data_split=='train'))


        elif self.data_source == 'TBD':
            # For other public timeseries-anomaly detection dataaset 
            raise NotImplementedError("Dataset for 'TBD' is not implemented yet.")

        else:
            raise ValueError(f"Unsupported data_source: {self.data_source}")

        return dataloader
    


class MatFolderDataset(Dataset):
    def __init__(self, folder_path, data_source, sensor_modality='lidar', seq_len_pred=None, min_obj_length=1, useEgoCentricCoord=False,  input_features_for_model=[],
                 data_type="normal", framework_task='', error_model_parameters={}, output_dir=None):
        assert framework_task in ['detection_of_injected_anomalies'], "unkown framework task"
        assert data_type in ['normal','normal_and_anomalies'], "unkown data_type"
        self.framework_task         = framework_task
        self.data_type              = data_type
        self.error_model_parameters = error_model_parameters
        self.output_dir             = output_dir
                
        self.data_source              = data_source
        self.folder_path              = folder_path
        self.seq_len_pred             = seq_len_pred
        self.min_obj_length           = min_obj_length
        self.useEgoCentricCoord       = useEgoCentricCoord
        self.sensor_modality          = sensor_modality
        self.output_dir               = output_dir
        self.input_features_for_model = input_features_for_model
        self.obj_data_order           = ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'pred_class', 'v', 'v_x', 'v_y', 'tracking_score']
        self.files                    = [os.path.join(folder_path, f)  for f in os.listdir(folder_path)  if f.endswith('.mat')]
        

        if self.framework_task == 'detection_of_injected_anomalies':
            if self.error_model_parameters['error_model'] == "IndependentErrorModel":
                self.error_model = IndependentErrorModel(error_params=error_model_parameters['error_statistics'], 
                                                        obj_features_to_alter=error_model_parameters['obj_features_to_alter'],
                                                        n_times_anomaly=error_model_parameters['n_times_anomaly'],
                                                        error_injected_in=error_model_parameters['error_injected_in'],
                                                        max_prediction_horizon=self.seq_len_pred,
                                                        output_dir=self.output_dir
                                                        )

            elif self.error_model_type == "PEM":
                self.error_model = PEM(error_params=error_model_parameters['error_statistics'], 
                                    obj_features_to_alter=error_model_parameters['obj_features_to_alter'],
                                    n_times_anomaly=error_model_parameters['n_times_anomaly'],
                                    max_prediction_horizon=self.seq_len_pred,
                                    output_dir=self.output_dir
                                    )

    def __len__(self):
        return len(self.files)
    

    def normalize_features(self, tensor):
        # tensor: [seq_len, num_features] or [batch, seq_len, num_features]
        return (tensor - self.feature_means) / (self.feature_stds + 1e-8)

    def __getitem__(self, idx):
        filename = self.files[idx]

        ### Process data from self.data_source
        if self.data_source == 'automotive_scene_level':
            # The self.min_obj_length is an important parameter here, as a minimal length is required to have a meaningful prediction horizon
            sample_scene = _load_scene_level_data(filename,
                                                  useEgoCentricCoord=True, 
                                                  min_obj_length=self.min_obj_length)

            # Decide which object list to process
            obj_ego = sample_scene['obj_ego']
            if self.sensor_modality == 'lidar':
                obj_list = sample_scene['obj_list_lidar']
            elif self.sensor_modality == 'camera':
                obj_list = sample_scene['obj_list_camera']
            else:
                raise ValueError(f"Unsupported sensor modality: {self.sensor_modality}")

        elif self.data_source == 'automotive_object_level':
            pass
            # Decide which object list to process
            # Arrange data


        ### Split data into past and future
        objects_history = []
        objects_future = []
        objects_label = []
        objects_whole = []
        #= obj_list[:, :self.seq_len_pred, :]
        for obj in obj_list:
            # objects_past of variable length: t = [0, -self.seq_len_pred]
            objects_history.append(torch.tensor(obj[:,:-self.seq_len_pred], dtype=torch.float32).transpose(0,1))
            # objects_future of fixed length: t = [-self.seq_len_pred: end]
            objects_future.append(torch.tensor(obj[:,-self.seq_len_pred:], dtype=torch.float32).transpose(0,1))
            assert (len(objects_future[-1]) + len(objects_history[-1])) == obj.shape[1], "Error, the past and future must sum up to the overall object length"
            objects_whole.append(torch.tensor(obj, dtype=torch.float32).transpose(0,1))
        

        if self.framework_task == "detection_of_injected_anomalies":     
            # Application of PEM / fault injection       
            
            if self.data_type == "normal":      
                # Train set
                # No application of PEM / fault injection -> just take the normal data         
                objects_label.extend([0]*len(objects_future))
            
            elif self.data_type == "normal_and_anomalies":
                # Test/Val-(Evaluation) Phase contains normal and perturbed data 
                if self.error_model_parameters['error_injected_in'] == "future":                  
                    # Future
                    objects_future_errors = self.error_model.generate_errorous_objects(original_objects=objects_future)
                    
                    objects_label   = [0] * len(objects_future) + [1] * len(objects_future_errors)
                    objects_future  = objects_future  + objects_future_errors
                    objects_history = objects_history + objects_history
                    objects_whole   = objects_whole   + objects_whole
            
                elif self.error_model_parameters['error_injected_in'] == "history":                
                    # History
                    objects_history_errors = self.error_model.generate_errorous_objects(original_objects=objects_history)
                    
                    objects_label   = [0] * len(objects_history) + [1] * len(objects_history_errors)
                    objects_future  = objects_future  + objects_future
                    objects_history = objects_history + objects_history_errors
                    objects_whole   = objects_whole   + objects_whole
                
                elif self.error_model_parameters['error_injected_in'] == "whole_object":
                    # Whole Object
                    objects_whole_errors  = self.error_model.generate_errorous_objects(original_objects=objects_whole)                    
                    
                    objects_label   = [0] * len(objects_whole) + [1] * len(objects_whole_errors)
                    objects_future  = objects_future  + objects_future
                    objects_history = objects_history + objects_history
                    objects_whole   = objects_whole   + objects_whole_errors
       
                assert len(objects_whole) == len(objects_label) == len(objects_future) == len(objects_history), "For each object there must be a label"

        
        if len(objects_whole) > 0:
            ### Filter object features
            objects_history_filtered = filter_object_features(input_data=objects_history, 
                                                            data_order=self.obj_data_order,
                                                            features_to_keep=self.input_features_for_model)
            objects_future_filtered  = filter_object_features(input_data=objects_future, 
                                                            data_order=self.obj_data_order,
                                                            features_to_keep=self.input_features_for_model)
            objects_whole_filtered   = filter_object_features(input_data=objects_whole, 
                                                            data_order=self.obj_data_order,
                                                            features_to_keep=self.input_features_for_model)


            ### Convert to tensors
            objects_label       = torch.tensor(objects_label)
            # Variable lengths with padding 
            objects_history_padded  = torch.nn.utils.rnn.pad_sequence(objects_history_filtered, batch_first=True)  # Padding requires 
            objects_whole_padded = torch.nn.utils.rnn.pad_sequence(objects_whole_filtered, batch_first=True)  # Padding requires 
            # Fixed temporal length (as the prediction horizon is set by self.seq_len_pred)
            objects_future_stacked = torch.stack(objects_future_filtered)
            # Fixed temporal length within a scene
            obj_ego         = torch.tensor(sample_scene['obj_ego'], dtype=torch.float32)

            

            assert objects_history_padded.shape[0] == objects_future_stacked.shape[0] == objects_label.shape[0], "Must contain the same number of smaples"   
            assert objects_history_padded.shape[2] == objects_future_stacked.shape[2], "Must contain the same number of features"

        else:
            objects_history_padded = []
            objects_future_stacked = []
            objects_label          = []
            objects_whole_padded   = []


        out_dict = {'objects': {'past':   objects_history_padded,
                                'future': objects_future_stacked,
                                'label':  objects_label,
                                'whole':  objects_whole_padded,},
                    'obj_ego': obj_ego,}


        return out_dict