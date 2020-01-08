import os
import json
import glob

from PIL import Image
from feature.feature_defintions.Histogram import Histogram

# Constants
HISTOGRAM_BINS = 256

# Logging
#
# Flow:
#     1. Read input JSON
#     2. Instantiate feature object
#     3. Apply get_feature() to each image
#     4. Write each feature to CSV
#
# Other:
#     * Image/open mode on input JSON
#     * Convert all images to same format so can use one image package
#
# Feature object:
#     * Get feature method
#
# Input JSON structure:
# {
#     "data_path_base": "/home/kane/Datasets/",
#     "data_name": "eurosat/small/",
#     "index_path_base": "/home/kane/Indexes/",
#     "index_name": "check250.csv",
#     "feature_path": "/home/kane/Features/",
#     "feature_name": "vae"
# }


# Parse config file
def json_read():
    config_path = os.path.join(os.getcwd(), 'params.json')
    assert (os.path.isfile(config_path))

    with open(config_path) as f:
        config_file = json.load(f)
        input_path = config_file['data_path_base']
        input_name = config_file['data_name']
        output_path = config_file['index_path_base']
        output_name = config_file['index_name']
        feature_path = config_file['feature_path_base']
        feature_name = config_file['feature_name']

    config_dict = dict()
    config_dict['input'] = input_path + input_name
    config_dict['output'] = output_path + output_name
    config_dict['feature_name'] = feature_name
    config_dict['feature_path'] = feature_path

    return config_dict


# Instantiate object for requested feature
def feature_object(config):
    feature_type = config['feature_name']
    feature_path = config['feature_path']

    if "histogram" in feature_type:
        _feature = Histogram(HISTOGRAM_BINS, feature_type)
    else:
        print("not histogram")
        _feature = None

    return _feature


# Iterate through all images and write feature to CSV for each one
def index_create(config, feature):
    output_path = config['output']
    data_path = config['input']
    with open(output_path, "w") as f:
        cnt = 0
        for path in glob.glob(data_path + "*.jpg"):
            img_id = path[path.rfind("/") + 1:]
            image = Image.open(path)
            feature_vector = feature.get_feature(image)
            feature_vector = [str(f) for f in feature_vector]

            # Write each feature to CSV
            _feature = "{},{}\n".format(img_id[:-4], ",".join(feature_vector))
            f.write(_feature)
            cnt += 1
            if cnt % 100 == 0:
                print("Feature file: {}, idx: {}".format(img_id, cnt))


# Feature index creation code entry point called from top level main function
def feature_driver_run():
    # Read input JSON
    config = json_read()

    # Instantiate feature object
    feature = feature_object(config)

    # Determine image format in given directory

    # Convert all images to JPG

    # Apply get_feature() to each image
    index_create(config, feature)
