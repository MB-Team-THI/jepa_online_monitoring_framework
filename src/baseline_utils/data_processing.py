import copy
import random
import numpy as np


def covert_to_ego_centric_coordinate_system(data_in, data_order):
    # The ego object should be at the cooridinate origin at all time steps
    objects_in = data_in['objects']
    ego_in     = data_in['obj_ego']
    if ego_in == []:
        assert False, "EGO data must be present to convert to ego centric coordinate system"
    objects_out = []
    for obj, ego in zip(objects_in, ego_in):
        if type(obj) == dict:
            obj_idx = [int(x) for x in obj['time_idx_in_scenario_frame']]
            obj_x = obj['x']
            obj_y = obj['y']
            ego_x = np.array(ego['x'])[obj_idx]
            ego_y = np.array(ego['y'])[obj_idx]
        else:
            obj_idx = [int(x) for x in obj[:, data_order.index('time_idx_in_scenario_frame')]]
            obj_x   = obj[:, data_order.index('x')]
            obj_y   = obj[:, data_order.index('y')]        
            ego_x   = np.array(ego[:, data_order.index('x')])[obj_idx]
            ego_y   = np.array(ego[:, data_order.index('y')])[obj_idx]

        obj_new = copy.deepcopy(obj)
        obj_new['x'] = (obj_x - ego_x).tolist()
        obj_new['y'] = (obj_y - ego_y).tolist()
        objects_out.append(obj_new)

    return {'objects': objects_out,
            'obj_ego': ego_in,}



def filter_object_part(input_objects, error_injected_in, T_fut):
    if error_injected_in == "whole_object":
        output_objects = input_objects
    else:
        output_objects = []      
        for obj in input_objects:
            
            if error_injected_in == "history":
                obj_hist = {k: obj[k][:-T_fut] for k in obj}
                output_objects.append(obj_hist)

            elif error_injected_in == "future":                
                obj_hist = {k: obj[k][-T_fut:] for k in obj}
                output_objects.append(obj_hist)

    return output_objects


def filter_object_length(data_in, min_length):
    objects_in = data_in['objects']
    ego_in     = data_in['obj_ego']
    if ego_in != []:
        assert len(objects_in) == len(ego_in), "object list and ego list must be of same length"

    objects_out, ego_out = [], []
    first_key = list(objects_in[0].keys())[0]
    for idx, obj in enumerate(objects_in):
        if len(obj[first_key]) >= min_length:
            objects_out.append(obj)
            if ego_in != []:
                ego_out.append(ego_in[idx])
    
    if ego_in != []:
        assert len(objects_out) == len(ego_out), "object list and ego list must be of same length"

    return {'objects': objects_out, 
            'obj_ego': ego_out}


def filter_relevant_features(obj_list, relevant_features):
    # Filter the relevant features
    for obj in obj_list:
        for key in list(obj.keys()):
            if key not in relevant_features:
                obj.pop(key)
    return obj_list


def cut_objects_to_same_length(obj_list_in, n):
    # Bring objects to same length and flatten them 2D
    obj_list_out = []
    for obj in obj_list_in:
       obj_list_out.append(obj[:, 0:n].flatten())

    return obj_list_out


def same_length_padding(input_data, goal_length, padding_value=0, flat_objects=True):
    # Pad and cut objects + flat objects
    output_data = []
    for obj in input_data:
        obj_length = obj.shape[1]
        if goal_length < obj_length:
            # Cut object to goal length
            obj_new = [x[:goal_length] for x in obj]
        elif obj_length < goal_length:
            # Pad until goal length
            vals = [padding_value] * (goal_length-obj_length)
            obj_new = [np.concatenate((x, vals), axis=0) for x in obj]
        else:
            # Desired length - do nothing
            obj_new = obj
        if flat_objects:
            obj_new = np.array(obj_new).flatten()
        output_data.append(obj_new)

    return output_data


def bring_objects_to_same_length(input_data, params):
    method = params['same_length_method']
    if params['data_format'] == 'object_pairs':   
        if params['obj_pair_rep'] == 'diff':
            goal_length = params['goal_length']
        else:
            # Obj Pair representation: c+l or c+diff
            goal_length = params['goal_length'] * 2
    else:
        goal_length = params['goal_length']

    if method == "padding":
        output_data = same_length_padding(input_data, goal_length=goal_length)

    elif method == "cutting":
        output_data = cut_objects_to_same_length(input_data, n=goal_length)

    return output_data


def take_future_part(input_data, t_fut, flat_objects=True):
    # Only take the future part of the embeddings to make the task for the baseline easier
    output_data = []
    for obj in input_data:
        obj_future = obj[:, -t_fut:]

        if flat_objects:
            obj_future = np.array(obj_future).flatten()
        output_data.append(obj_future)
    
    output_data = np.array(output_data)

    return output_data

def take_history_part(input_data, t_fut):
    # Only take the history part of the embeddings to make the task for the baseline easier
    output_data = []
    for obj in input_data:
        obj_history = obj[:, :-t_fut]

        output_data.append(obj_history)

    return output_data


def arrange_embeddings(data_in, params):
    z_dim_ad = params['z_dim_ad']

    ### Embedding Pairs
    res_obj_pairs = []
    for emb_pair in data_in:

        # Partition latent space
        if params['partition_space']:
            obj_camera = np.array(emb_pair['obj_camera'][z_dim_ad:])
            obj_lidar  = np.array(emb_pair['obj_lidar'][z_dim_ad:])
        else:
            obj_camera = np.array(emb_pair['obj_camera'])
            obj_lidar  = np.array(emb_pair['obj_lidar'])
    
        ### Rearange the embeddings for anomaly detection
        if params['arrangement_method'] == 'diff':
            emb_pair_arranged = obj_camera - obj_lidar #  [(x[0]-x[1]).tolist() for x in emb_pairs]
                
        elif params['arrangement_method'] == 'concat':
            emb_pair_arranged = np.concatenate((obj_camera, obj_lidar)) # [torch.cat((x[0], x[1])).tolist() for x in emb_pairs]
                    
        elif params['arrangement_method'] == 'concat+diff':
            emb_pair_arranged = np.concatenate((obj_camera, obj_lidar, obj_camera-obj_lidar)) # [torch.cat((x[0], x[1], x[0] - x[1] )).tolist() for x in emb_pairs]
                
        elif params['arrangement_method'] == 'camera+diff':
            emb_pair_arranged = np.concatenate((obj_camera, obj_camera-obj_lidar))   # [torch.cat((x[0], x[0] - x[1])).tolist() for x in emb_pairs]

        else:
            assert False, "Unkown params['arrangement_method']"

        res_obj_pairs.append(emb_pair_arranged)

    
    res_obj_pairs=np.array(res_obj_pairs)
    return res_obj_pairs


def get_of_object_pair_representation(obj_pair_list, representation_type='diff'):
    # Calculate the differences within an object pair 'x','y', 'z','size_x','size_y', 'size_z', 'heading', 'v_x', 'v_y', 'v'
    relevant_features = ['x','y', 'z', 'heading', 'v_x', 'v_y', 'v', 'size_x','size_y', 'size_z', 'tracking_score']
    obj_pair_keys = list(obj_pair_list[0].keys())
    key_1 = obj_pair_keys[0]    # Camera key
    key_2 = obj_pair_keys[1]    # LiDAR key
    obj_pair_rep_list = []
    for obj_pair in obj_pair_list:
        time_idx_1 = obj_pair[key_1]['time_idx_in_scenario_frame']
        time_idx_2 = obj_pair[key_2]['time_idx_in_scenario_frame']
        joint_time_idx = [x for x in time_idx_1 if x in time_idx_2]

        # Only consider the temporal overlap between the objects
        obj_representation = {}
        for key in obj_pair[key_1]:
            if key in relevant_features:
                if representation_type == 'c_l':
                    # Concat camera and lidar object
                    obj_representation[key] = obj_pair[key_1][key] + obj_pair[key_2][key]

                elif representation_type == 'diff':
                    obj_representation[key] = [obj_pair[key_1][key][time_idx_1.index(v)] - obj_pair[key_2][key][time_idx_2.index(v)] for v in joint_time_idx]
                
                elif representation_type == 'c_diff':
                    diff  = [obj_pair[key_1][key][time_idx_1.index(v)] - obj_pair[key_2][key][time_idx_2.index(v)] for v in joint_time_idx]
                    obj_representation[key] = obj_pair[key_1][key] + diff
                    
                elif representation_type == 'c_l_diff':
                    diff  = [obj_pair[key_1][key][time_idx_1.index(v)] - obj_pair[key_2][key][time_idx_2.index(v)] for v in joint_time_idx]
                    obj_representation[key] = obj_pair[key_1][key] + obj_pair[key_2][key] + diff
   
                elif representation_type == 'mean_of_single_feature_diff':
                    obj_representation[key] = [np.mean(obj_pair[key_1][key]) - np.mean(obj_pair[key_2][key])]
                else:
                    assert False, "uknown representation type: " + representation_type

        # Add time idx
        if representation_type == 'c_l':
            c_l_time_idx = time_idx_1 + time_idx_2
            first_time_idx = min(c_l_time_idx)
            obj_representation['time_idx_in_scenario_frame'] = [x-first_time_idx for x in c_l_time_idx]

        elif representation_type == 'diff':
            first_time_idx = min(joint_time_idx)
            obj_representation['time_idx_in_scenario_frame'] = [x-first_time_idx for x in joint_time_idx]

        elif representation_type == 'c_diff':
            c_diff_time_idx = time_idx_1 + joint_time_idx
            first_time_idx = min(time_idx_1)
            obj_representation['time_idx_in_scenario_frame'] =  [x-first_time_idx for x in c_diff_time_idx]

        assert len(obj_representation['time_idx_in_scenario_frame'])==len(obj_representation['x']), "must be of same length"

        obj_pair_rep_list.append(obj_representation)
    return obj_pair_rep_list


def process_object_list_for_anomaly_detection(obj_list_input, key='', params={}, data_split="train_normal"):
    input_for_ad = params['input_for_ad']
    assert input_for_ad in ['history', 'future', 'whole_object'], "Undefined AD input"

    ### Filter object features
    object_list_filtered = filter_relevant_features(obj_list_input, params['filter_relevant_features'])

    ### Toy example
    if params['toy_data']['enabled']:        
        # Do not use the complete dataset yet -> only n_toy_datas data samples
        random.seed(42)
        n_toy_data  = params['toy_data']['n_samples']
        toy_indices = random.sample(range(0, len(object_list_filtered)-1), n_toy_data)
        object_list_filtered = np.array(object_list_filtered)[toy_indices]
    
    ### Convert from dict to numpy.ndarray
    object_list_array = [np.array([np.array(v) for v in sample.values()]) for sample in object_list_filtered]
        
    ### Bring to equal length and flat them
    if input_for_ad in ['history', 'whole_object']:
        object_list_same_length = bring_objects_to_same_length(object_list_array, params)
        
    elif input_for_ad == 'future':
        # As T_fut is constant there already of same length -> now only
        object_list_same_length = [ np.array(obj).flatten() for obj in object_list_array]

    return object_list_same_length



def rot_points(xy,angle):
    rotMat = np.array([[np.cos(angle),-np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    points = np.matmul(rotMat ,xy)
    return points


def convert_obj(obj, ego):
    obj_new = obj.copy()
    # Get EGO data for obj timesteps
    obj_time_idx = [int(x) for x in obj['time_idx_in_scenario_frame']]
    ego_x = np.array(ego['x'])[obj_time_idx]
    ego_y = np.array(ego['y'])[obj_time_idx]
    ego_heading = np.array(ego['heading'])[obj_time_idx]

    # Center to EGO-position
    x_new = obj['x'] - ego_x
    y_new = obj['y'] - ego_y

    # Rotate around EGO heading
    heading_new = []
    for t_idx in range(len(x_new)):
        angle = ego_heading[t_idx]
        pos_t = [x_new[t_idx], y_new[t_idx]]
        x_new[t_idx], y_new[t_idx] = rot_points(pos_t, angle)
        heading_new.append(- obj['heading'][t_idx] + angle)
                
    obj_new['x'] = x_new.tolist()
    obj_new['y'] = y_new.tolist()
    obj_new['heading'] = heading_new

    return obj_new


def convert_to_ego_centric(dataset, keys_to_convert=['obj_pair_camera_lidar_normal']):

    for key in keys_to_convert:
        obj_pair_in = copy.deepcopy(dataset['object_pairs']['data'][key])
        obj_pairs_out = []

        for obj_pair in obj_pair_in:
            obj_c   = copy.deepcopy(obj_pair['obj_camera'])
            obj_l   = copy.deepcopy(obj_pair['obj_lidar'])
            obj_ego = obj_pair['obj_ego']

            obj_c_new = convert_obj(obj=obj_c, ego=obj_ego)
            obj_l_new = convert_obj(obj=obj_l, ego=obj_ego)


            obj_pair_new = {'obj_camera':   obj_c_new,
                            'obj_lidar':    obj_l_new,
                            'obj_ego':      obj_ego,}
            obj_pairs_out.append(obj_pair_new)


        dataset['object_pairs']['data'][key] = obj_pairs_out

    return dataset