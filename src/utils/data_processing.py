

def filter_object_features(input_data, data_order, features_to_keep):
    
    assert [feature_name in data_order for feature_name in features_to_keep], "All features to keep must be in the data order"
    assert len(features_to_keep) == len(set(features_to_keep)), "Features to keep must be unique"
    
    idx_to_keep = [data_order.index(feature_name) for feature_name in features_to_keep ]

    output_data = []
    for obj in input_data:
        assert len(data_order) == obj.shape[1], "Data order length must match input data shape"
        # Keep all time steps and only filter the features
        output_data.append(obj[:, idx_to_keep])

    assert len(features_to_keep) == output_data[0].shape[1], "Number of features to keep must match the output data shape"
    

    return output_data
