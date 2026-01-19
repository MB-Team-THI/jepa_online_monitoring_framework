
import numpy as np
from scipy.io import loadmat

from src.dataset.loaders.helpers import get_obj_list_associated_format
from src.dataset.loaders.helpers import _dict_to_ndarray
from src.utils.rotate_and_align_traj import convert_to_ego_centric

# Load a single sample from the dataset

def _load_scene_level_data(filename, keys=None, label_key=None, useEgoCentricCoord=False, min_obj_length=1):
    # Load object data from a .mat file
    all_keys_to_load = ["general_info", "scene_info", "obj_list_ego", "obj_list_lidar", "obj_list_camera", "obj_list_gt", "obj_list_lidar_nc", "association_list"]
    mat_temp = loadmat(filename, variable_names = all_keys_to_load, verify_compressed_data_integrity=False)

    # META===============================================================================================
    # INFO-----------------------------------------------------------------------------------------------

    ### scene_info
    scene_keys      = mat_temp['scene_info']['scene'][0][0][0].__dir__.__self__.dtype.names
    scene_vals      = [mat_temp['scene_info']['scene'][0][0][0][0][i][0][:] for i in range(len(scene_keys))]
    scene_vals[2]   = scene_vals[2][0]
    scene_dict      = {k: v for (k, v) in zip(scene_keys, scene_vals)}
    map_keys        = mat_temp['scene_info']['map'][0][0][0].__dir__.__self__.dtype.names
    map_vals        = [mat_temp['scene_info']['map'][0][0][0][0][i][0][:] for i in range(len(map_keys))]
    map_dict        = {k: v for (k, v ) in zip(map_keys, map_vals)}
    scene_info      = {'scene': scene_dict, 'map': map_dict}

    ### general_info
    data_order      = [x.strip() for x in mat_temp["general_info"][0]['data_order_tracking_res'][0]]
    class_names     = mat_temp["general_info"][0]['class_name_dict'][0]
    class_name_dict = {k: class_names[k][0][0][0][0] for k in class_names.__dir__.__self__.dtype.names}
    general_info    = {"data_order_tracking_res":  data_order,
                       "class_name_dict":          class_name_dict}

    # DYNAMICS============================================================================================
    # EGO obj list----------------------------------------------------------------------------------------
    obj_ego_dict: list = {k: mat_temp["obj_list_ego"][k][0][0][0] for k in mat_temp["obj_list_ego"].dtype.names}
    obj_ego = _dict_to_ndarray(obj_ego_dict, obj_ego_dict.keys())

    feature_keys = [] 
    obj_list_camera = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_camera", min_obj_length=min_obj_length)
    obj_list_lidar  = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_lidar",  min_obj_length=min_obj_length)
    obj_list_gt     = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_gt",     min_obj_length=min_obj_length)

    # Adapt coordinate system
    if useEgoCentricCoord:
        # Convert the camera and lidar object list
        obj_list_camera = convert_to_ego_centric(obj_list_in=obj_list_camera, ego_obj=obj_ego_dict, data_order_obj=data_order)
        obj_list_lidar  = convert_to_ego_centric(obj_list_in=obj_list_lidar,  ego_obj=obj_ego_dict, data_order_obj=data_order)
        data_order_gt = []
        # obj_list_gt     = convert_to_ego_centric(obj_list_in=obj_list_gt,     ego_obj=obj_ego_dict, data_order_obj=data_order_gt)
    else:
        # Stay with the global ego-centric coordinate system
        pass
    


    scene_dict = {"general_info":    general_info,
                  "scene_info":      scene_info,
                  "obj_ego":         obj_ego,
                  "obj_list_lidar":  obj_list_lidar,
                  "obj_list_camera": obj_list_camera,
                  "obj_list_gt":     obj_list_gt,}

    return scene_dict
