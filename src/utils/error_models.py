'''
PEM = Perception Error Models
based on Piazzoni 2024 "PEM: Perception Error Model for Virtual Testing of Autonomous Vehicles"
'''

import torch
import random
import numpy as np

import copy




class IndependentErrorModel():
    def __init__(self, error_params, obj_features_to_alter, n_times_anomaly=2, max_prediction_horizon=4, error_injected_in='future', output_dir=None):
        self.error_mean             = error_params['mean']
        self.error_variance         = error_params['variance']
        self.obj_features_to_alter  = obj_features_to_alter
        self.n_times_anomaly        = n_times_anomaly
        self.error_injected_in      = error_injected_in
        self.max_prediction_horizon = max_prediction_horizon
        self.output_dir             = output_dir
        self.obj_data_order         = ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'pred_class', 'v', 'v_x', 'v_y', 'tracking_score']


    def generate_errorous_objects(self, original_objects):
        
        errorous_objects = []
        for org_obj in original_objects:  

            if torch.is_tensor(org_obj):
                # Code for real model
                errorous_obj = org_obj.clone().detach()
                
                for feature_name in self.obj_features_to_alter:
                    errorous_timesteps = random.sample(range(0, self.max_prediction_horizon), self.n_times_anomaly)

                    for t in errorous_timesteps:
                        # Calculate the random error and add it up
                        error_abs = np.random.normal(loc=self.error_mean[feature_name], scale=self.error_variance[feature_name])
                        sign = random.choices([-1,1])[0]

                        errorous_obj[t][self.obj_data_order.index(feature_name)] += (error_abs * sign)

            else:
                # Code for baseline-run
                errorous_obj = copy.deepcopy(org_obj)
                obj_length = len(errorous_obj['x'])

                for feature_name in self.obj_features_to_alter:
                    errorous_timesteps = random.sample(range(0, obj_length), self.n_times_anomaly)

                    for t in errorous_timesteps:
                        # Calculate the random error and add it up
                        error_abs = np.random.normal(loc=self.error_mean[feature_name], scale=self.error_variance[feature_name])
                        sign = random.choices([-1,1])[0]

                        errorous_obj[feature_name][t] += (error_abs * sign)
            
            errorous_objects.append(errorous_obj)


        return errorous_objects
