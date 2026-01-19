'''
Author:     Alexander Fertig
Date:       2025-09-12
Purpose:    Apply standard anomaly detection methods on object lists in their input space (not encoded)
            Compare the anomaly deteciton performance on the object lists in the latent space and original space.
            
Steps:
    1. Get Data 
        - Get data and save it OR
        - Load object list data

    2. Prepare Data
        - Define task (detection of injected faults)
        - Perform this on the fly

    3. Anoamaly detection
        - apply anomaly detection mehtods: GMM, DBSCAN, ABOD, LOF, IsolationForest, COPOD
        - Visiualize the dim-reduced space via PCA, t-SNE and UMAP

    4. Save results
'''

import os
from datetime import date

from src.utils.save_json_files import load_json_file, save_json_file
from src.baseline_utils.framework_task import process_framework_tasks
from src.baseline_utils.data_processing import process_object_list_for_anomaly_detection
from src.utils.anomaly_detection_methods import perform_anomaly_detection


##################################
### Settings
today_str = str(date.today()).replace('-','')
folder_out = today_str + '_benchmark_anomaly_detection'
output_dir = os.path.join('result_folder', 'baselines', folder_out)

### Task
framework_tasks = ['detection_of_injected_anomalies']
task_settings = {'framework_task':      framework_tasks[0],
                 'obj_part_for_errormodel':      'history',                             # ['history', 'future', 'whole_object']
                 'embeddings_4_ad':     "max_token",                                    # ["max_token", "avg_token", "CLS_encoder", "Tokens_future", "last_token", "CLS_fut_vs_pred"]
                 'error_model_params':  {'error_model':       'IndependentErrorModel',  # ['IndependentErrorModel', 'PEM']
                                         'error_injected_in': '',                       # see ['obj_part_for_errormodel'] 
                                         'n_times_anomaly':   1,
                                         'error_statistics':  {'mean':       {'heading':0, 'v':5,},
                                                               'variance':   {'heading': 0.1, 'v': 0.1,},
                                                              },
                                         'obj_features_to_alter': ['v'],
                                        },
                 }
task_settings['error_model_params']['error_injected_in'] = task_settings['obj_part_for_errormodel']

### Data
use_single_objects = False
data_dir  = "data\\nuscenes\\baseline"

if use_single_objects:
    sensor_type = 'obj_list_lidar'      # 'obj_list_lidar' or 'obj_list_camera'
else:
    sensor_type = 'obj_lidar'           # 'obj_lidar' or 'obj_camera'   

### Data Processing
same_length_methods = ['dtw_resampling', 'dtw_distance', 'padding', 'cutting']
data_processing_params = {'data_format':                'single_objects_v2',
                          'filter_min_object_length':   8, 
                          # Available features: ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'pred_class', 'v', 'v_x', 'v_y', 'tracking_score']
                          'filter_relevant_features':   ['x', 'y', 'heading', 'v'],
                          'input_for_ad':               task_settings['error_model_params']['error_injected_in'],       # ['history', 'future', 'whole_object']
                          'same_length_method':         same_length_methods[2],
                          'goal_length':                40,
                          'toy_data':                   {'enabled': False},
                          'T_fut':                      4,
                          'obj_pair_rep':               'not_relevant',
                          'data_order':                 ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'pred_class', 'v', 'v_x', 'v_y', 'tracking_score'],
                          }

### Anomaly Detection Methods    
evaluation_settings = {'output_dir':  output_dir,
                       'data_source': sensor_type,
                       'anomaly_detection': 
                                {'lof':  {'enabled': True,
                                          'n_neighbors': 15,
                                          'novelty': True},
                                 'abod': {'enabled': True, 
                                          'method': 'unify'},
                                 'gmm':  {'enabled': True,
                                          'n_components': 5,
                                          'covariance_type': 'full',
                                          'log_likelihood_threshold_percentil': 5,},
                                 'output_dir': output_dir,
                                },
                      }

def main():
    ###############################
    ### Get Data
    print("--- Start Data Loading ---")

    # Load data which is already preprocessed
    if use_single_objects:

        filename_val   = os.path.join(data_dir, "val_set_single_objects.json")
        filename_test  = os.path.join(data_dir, "test_set_single_objects.json")
        filename_train = os.path.join(data_dir, "train_set_single_objects.json")
    else:
        filename_val   = os.path.join(data_dir, "val_set_all_objects.json")
        filename_test  = os.path.join(data_dir, "test_set_all_objects.json")
        filename_train = os.path.join(data_dir, "train_set_all_objects.json")

    data_test  = load_json_file(filename_test)
    data_train = load_json_file(filename_train)
    data_val   = load_json_file(filename=filename_val)

        
    if use_single_objects:
        input_train = {'objects': data_train['data'][sensor_type],
                       'obj_ego': []}
        input_test  = {'objects': data_test['data'][sensor_type],
                       'obj_ego': []}
        input_val   = {'objects': data_val['data'][sensor_type],
                       'obj_ego': []}
    else:
        input_train = {'objects': [x[sensor_type] for x in data_train['object_pairs']['data']['obj_pair_camera_lidar_normal']],
                       'obj_ego': [x['obj_ego']   for x in data_train['object_pairs']['data']['obj_pair_camera_lidar_normal']],}
        input_test  = {'objects': [x[sensor_type] for x in data_test['object_pairs']['data']['obj_pair_camera_lidar_normal']],
                       'obj_ego': [x['obj_ego']   for x in data_test['object_pairs']['data']['obj_pair_camera_lidar_normal']],}
        input_val   = {'objects': [x[sensor_type] for x in data_val['object_pairs']['data']['obj_pair_camera_lidar_normal']],
                       'obj_ego': [x['obj_ego']   for x in data_val['object_pairs']['data']['obj_pair_camera_lidar_normal']],}

    ###############################
    ### Process Object Lists based on Framework Task
    print("--- Start Data Processing ---")
    objects_train, labels_train = process_framework_tasks(input_data=input_train, task_settings=task_settings, key=sensor_type, data_processing_params=data_processing_params, data_split="train") 
    objects_test, labels_test   = process_framework_tasks(input_data=input_test,  task_settings=task_settings, key=sensor_type, data_processing_params=data_processing_params, data_split="test") 
    objects_val, labels_val     = process_framework_tasks(input_data=input_val,   task_settings=task_settings, key=sensor_type, data_processing_params=data_processing_params, data_split="val") 

    ###############################
    ### Data Processing
    objects_train_2 = process_object_list_for_anomaly_detection(obj_list_input=objects_train, key=sensor_type, params=data_processing_params)
    objects_test_2  = process_object_list_for_anomaly_detection(obj_list_input=objects_test,  key=sensor_type, params=data_processing_params)
    objects_val_2   = process_object_list_for_anomaly_detection(obj_list_input=objects_val,   key=sensor_type, params=data_processing_params)

    ###############################
    ### Anomaly Detection
    print("--- Start Anomaly Detection ---")
    ad_results = perform_anomaly_detection(x_train        = objects_train_2,
                                           x_test         = objects_test_2,
                                           y_test         = labels_test,
                                           settings       = evaluation_settings['anomaly_detection'],
                                           framework_task = task_settings['framework_task'])
    
    ###############################
    ### Save results
    res_dict = {'ad_results': ad_results,
                'params':  {'task_params':              task_settings,
                            'data_processing_params':   data_processing_params,
                            'evaluation_settings':      evaluation_settings,
                            },
                'info':    {'date':         str(date.today()),
                            'repository':   'jepa_approach',
                            'script':       'main_baseline.py',
                            'data_used':   {'filename_val':     filename_val,
                                            'filename_test':    filename_test,
                                            'filename_train':   filename_train,}, 
                            },           
                }
    save_json_file(output_dir=output_dir, 
                    filename="results.json", 
                    data=res_dict)
    
    print("--- Script completed ---")


if __name__ == '__main__':
    main()