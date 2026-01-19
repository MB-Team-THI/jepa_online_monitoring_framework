
from src.utils.error_models import IndependentErrorModel, PEM
from src.baseline_utils.data_processing import filter_object_length, filter_object_part, covert_to_ego_centric_coordinate_system




def setup_anomaly_injection(input_objects, params, data_type="normal"):
    error_model_type = params['error_model']
    
    if data_type == "normal":
        object_labels = [0] * len(input_objects)
        output_objects = input_objects
    
    elif data_type == "normal_and_anomalies":
        ### Setup Error Model
        if error_model_type == "IndependentErrorModel":
            error_model = IndependentErrorModel(error_params           = params['error_statistics'], 
                                                obj_features_to_alter  = params['obj_features_to_alter'],
                                                n_times_anomaly        = params['n_times_anomaly']
                                                )

        elif error_model_type == "PEM":
            error_model = PEM(error_params             = params['error_statistics'], 
                                obj_features_to_alter  = params['obj_features_to_alter'],
                                n_times_anomaly        = params['n_times_anomaly'],
                                max_prediction_horizon = 8
                                )
        else:
            assert False, "Unknown error_model_type:" + error_model_type

        ### Apply error model on input data
        
        # Application of PEM / fault injection
        objects_errors = error_model.generate_errorous_objects(original_objects=input_objects)

        # Double the dataset size: 50% normal objects and 50% perturbed objects
        object_labels   = [0] * len(input_objects) + [1] * len(objects_errors)
        output_objects  = input_objects  + objects_errors
        

    return output_objects, object_labels


def process_framework_tasks(input_data, task_settings, key, data_processing_params, data_split):

    min_obj_length = data_processing_params['filter_min_object_length']
    T_fut          = data_processing_params['T_fut']
    data_order     = data_processing_params['data_order']

    framework_task = task_settings['framework_task']
    task_settings['error_model_params']['error_injected_in'] = task_settings['obj_part_for_errormodel']
    error_injected_in = task_settings['error_model_params']['error_injected_in']
    
    assert error_injected_in in ["history", "future", "whole_object"]
    assert framework_task in ["detection_of_injected_anomalies"]


    if data_split == "train":
        # Train set should contain only normal data
        data_type = "normal"
    else:
        # Train set should contain only normal data
        data_type = "normal_and_anomalies"


    ### Filter based on object length
    objects_filtered_length = filter_object_length(input_data, min_obj_length)
    

    ### Filter relevant Object part
    objects_filtered_part = filter_object_part(input_objects     = objects_filtered_length['objects'],
                                               error_injected_in = error_injected_in,
                                               T_fut             = T_fut)
    

    ### Framework Task
    if framework_task == "detection_of_injected_anomalies":
        output_objects, object_labels = setup_anomaly_injection(input_objects = objects_filtered_part, 
                                                                params        = task_settings["error_model_params"], 
                                                                data_type     = data_type)

    
    assert len(output_objects) == len(object_labels), "For each object there must be a label"


    return output_objects, object_labels
