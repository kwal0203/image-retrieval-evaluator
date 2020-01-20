from json import load
from os import path, getcwd
from feature.feature_defintions.Histogram import Histogram
from feature.feature_defintions.AlexNet import AlexNet
from feature.feature_defintions.VGGNet import VGGNet
from feature.feature_defintions.ResNet50 import ResNet50
from feature.feature_defintions.Colourization import Colourization
from feature.feature_defintions.VAE import VAE
from feature.feature_tests.test_functions import *

# TODO:
#   Logging
#   Timing script
#   Output multiple CSV files for 1 run (i.e. models trained to different
#   epochs)

# Constants
HISTOGRAM_BINS = 256


# Parse feature config JSON file. Config structure:
#
# {
#     "input_path_base": "path/to/input/data/",
#     "input_name": "data_directory_name/",
#     "output_path_base": "path/to/output/",
#     "output_name": "name_of_index.csv",
#     "feature_path": "path/to/model/", (leave blank for histogram features)
#     "feature_name": "name_of_feature"
# }
def feature_json_read():
    config_path = path.join(getcwd(), 'feature_params.json')
    assert(path.isfile(config_path)), "{} does not exist".format(config_path)
    print("Feature config file path: ", config_path)

    with open(config_path) as f:
        config_file = load(f)
        input_path = config_file['input_path_base']
        input_name = config_file['input_name']
        output_path = config_file['output_path_base']
        output_name = config_file['output_name']
        feature_path = config_file['feature_path']
        feature_name = config_file['feature_name']
        layer = config_file['layer']

    config_dict = dict()
    config_dict['input'] = input_path + input_name
    config_dict['output'] = output_path + output_name
    config_dict['feature_name'] = feature_name
    config_dict['feature_path'] = feature_path
    config_dict['layer'] = layer

    return config_dict


# Instantiate object for requested feature
def feature_object_create(config):
    feature_name = config['feature_name']

    print(feature_name + " selected")
    if 'histogram' in feature_name:
        _feature = Histogram(HISTOGRAM_BINS, config)
    elif 'alex' in feature_name:
        _feature = AlexNet(config)
    elif 'vgg' in feature_name:
        _feature = VGGNet(config)
    elif 'resnet50' in feature_name:
        _feature = ResNet50(config)
    elif 'colourization' in feature_name:
        _feature = Colourization(config)
    elif 'vae' in feature_name:
        _feature = VAE(config)
    else:
        print("No feature selected")
        _feature = None

    return _feature


# Feature index creation code entry point called from top level main function
def feature_driver_run():
    # Read input JSON
    config = feature_json_read()
    test_config_dict(config)

    # Instantiate feature object
    feature_object = feature_object_create(config)
    # test_histogram_object(feature_object)

    # Apply get_feature() to each image
    feature_object.index_create(feature=feature_object)

