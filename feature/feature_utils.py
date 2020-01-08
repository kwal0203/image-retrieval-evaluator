import os
from feature.feature_defintions.Histogram import Histogram


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


def json_read():
    config_path = os.path.join(os.getcwd(), 'params.json')
    assert(os.path.isfile(config_path))
    return config_path


def feature_create():
    print("feature_create() not implemented")


def feature_driver_run():
    # Read input JSON
    config = json_read()

    # Instantiate feature object


    # 3. Apply get_feature() to each image
    # 4. Write each feature to CSV
