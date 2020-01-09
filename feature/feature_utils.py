from json import load
from skimage import io
from os import listdir, path, getcwd
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


def test_config_dict(config_dict):
    print("----- TEST CONFIG DICT -----")
    print("input:        ", config_dict['input'])
    print("output:       ", config_dict['output'])
    print("feature_name: ", config_dict['feature_name'])
    print("feature_path: ", config_dict['feature_path'])
    print("--- END TEST CONFIG DICT ---")


def test_histogram_object(histogram_object):
    print("----- TEST HISTOGRAM OBJECT -----")
    print("Histogram name: ", histogram_object.histogram_type)
    print("Histogram bins: ", histogram_object.bins)
    print("--- END TEST HISTOGRAM OBJECT ---")


# Parse config file
def json_read():
    config_path = path.join(getcwd(), 'feature_params.json')
    assert(path.isfile(config_path)), "{} does not exist".format(config_path)
    print("Config file path: ", config_path)

    with open(config_path) as f:
        config_file = load(f)
        input_path = config_file['input_path_base']
        input_name = config_file['input_name']
        output_path = config_file['output_path_base']
        output_name = config_file['output_name']
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
    feature_name = config['feature_name']
    feature_path = config['feature_path']

    print(feature_name + " selected")
    if "histogram" in feature_name:
        _feature = Histogram(HISTOGRAM_BINS, feature_name)
    else:
        _feature = None

    return _feature


# Iterate through all images and write feature to CSV for each one
def index_create(config, feature):
    output_path = config['output']
    input_path = config['input']
    assert(not path.isfile(output_path)), "{} exists".format(output_path)

    with open(output_path, "w+") as f:
        cnt = 0

        for filename in listdir(input_path):
            # Filename without file extension used as name of feature in CSV
            img_id = filename[:-4]

            # Calculate feature vector
            img = io.imread(input_path + filename)
            feature_vector = feature.get_feature(img)
            feature_vector = [str(f) for f in feature_vector]

            # Turn feature into string and write it to CSV file
            _feature = "{},{}\n".format(img_id, ",".join(feature_vector))
            f.write(_feature)
            cnt += 1
            if cnt % 1 == 0:
                print("Feature file: {}, idx: {}".format(img_id, cnt))


# Feature index creation code entry point called from top level main function
def feature_driver_run():
    # Read input JSON
    config = json_read()
    test_config_dict(config)

    # Instantiate feature object
    feature = feature_object(config)
    test_histogram_object(feature)

    # Apply get_feature() to each image
    index_create(config, feature)
