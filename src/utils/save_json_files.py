
import os
import json


def save_json_file(output_dir, filename, data):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename_dir = os.path.join(output_dir, filename)
    with open(filename_dir, 'w') as f:
        json.dump(data, f)



def load_json_file(filename):
    f = open(filename)
    loaded_json_file = json.load(f)
    f.close()

    return loaded_json_file
