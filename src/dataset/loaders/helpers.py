import os
import copy
import numpy as np


def _dict_to_ndarray(obj_dict, features_to_load):
    dict_keys = list(obj_dict.keys())

    obj_array = np.ndarray(shape=(len(features_to_load), len(obj_dict[dict_keys[0]])))

    idx = 0
    for key in features_to_load:
        if key in dict_keys:
            obj_array[idx, :] = obj_dict[key]
            idx += 1

    return obj_array


def get_obj_list_associated_format(data_in, object_list_key, feature_keys=None, min_obj_length=1):
    # Get object list from scene-format (of already associated objects)
    obj_list_out = []
    if object_list_key in data_in:
        if len(data_in[object_list_key]) > 0:            
            # Obtain the data from the nested .mat-file format
            object_list = []
            for obj in data_in[object_list_key][0]:
                if min_obj_length <= len(obj[0][obj[0].dtype.names[0]][0][0]):
                    obj_trans = {k: obj[0][k][0][0] for k in obj[0].dtype.names}
                    object_list.append(obj_trans)

            if len(object_list) > 0:
                # Filter based on feature_keys and convert to nd.array format (leave the intermediate dict-format for interpretability)
                if feature_keys==None:
                    # Take all keys
                    feature_keys = object_list[0].keys()
                for obj in object_list:
                    obj_list_out.append(_dict_to_ndarray(obj, feature_keys))

    return obj_list_out


def get_obj_list(data, data_order_key, class_name_dict=None):
    # convert the imported object data from the mat-files into one dict per object
    # The dict key describes the feature, while the values are strictly numerical (else they cannot be converted into tensors)

    obj_list            = []
    data_order_gt       = [i[0] for i in data['meta'][0][0][0][data_order_key][0][0]]

    if data.size != 0 and 0<len(data['results'][0][0]):
        for obj in data['results'][0][0][0]:
            complete_object = {}
            # Copy the complete obj data based on the data order
            for idx, key in enumerate(data_order_gt):
                complete_object[key] = obj[idx, :]
            # handle the index conversion from matlab to python
            complete_object['time_idx_in_scenario_frame'] = complete_object['time_idx_in_scenario_frame'] - 1        
            obj_list.append(complete_object)
    
    return obj_list

