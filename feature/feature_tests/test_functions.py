# Test code for feature functionality
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
