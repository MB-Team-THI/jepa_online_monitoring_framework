
import numpy as np
import matplotlib.pyplot as plt




def rot_points(xy,angle):
    rotMat = np.array([[np.cos(angle),-np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    points = np.matmul(rotMat ,xy)
    return points


def convert_to_ego_centric(obj_list_in, ego_obj, data_order_obj):
    # Convert the input obj list into an EGO-centric coordinate system
    # Here for every timestep the EGO-vehicle is in the center of the coordinate system and aligned to the right
    # Trajectories may only be plotted in a meaningful way with the positional data (x, y, heading) of the EGO 
    obj_list_out  = []
    idx_time_idx = data_order_obj.index('time_idx_in_scenario_frame')
    idx_x = data_order_obj.index('x')
    idx_y = data_order_obj.index('y')
    idx_heading = data_order_obj.index('heading')

    for obj_idx, obj in enumerate(obj_list_in): 
        obj_new = obj.copy()

        # Get EGO data for obj timesteps
        obj_time_idx = obj[idx_time_idx, :].astype(np.int8)
        ego_x = ego_obj['x'][obj_time_idx]
        ego_y = ego_obj['y'][obj_time_idx]
        ego_heading = ego_obj['heading'][obj_time_idx]

        # Center to EGO-position
        x_new = obj[idx_x, :] - ego_x
        y_new = obj[idx_y, :] - ego_y

        # Rotate around EGO heading
        heading_new = []
        for t_idx in range(len(x_new)):
            angle = ego_heading[t_idx]
            pos_t = [x_new[t_idx], y_new[t_idx]]
            x_new[t_idx], y_new[t_idx] = rot_points(pos_t, angle)
            heading_new.append(- obj[idx_heading, t_idx] + angle)
        
        obj_new[idx_x,:] = x_new
        obj_new[idx_y,:] = y_new
        obj_new[idx_heading,:] = np.array(heading_new)

        obj_list_out.append(obj_new)

    for i in range(len(obj_list_in)):
        assert obj_list_in[i].shape == obj_list_out[i].shape, "Must have the same shape"
    return obj_list_out
