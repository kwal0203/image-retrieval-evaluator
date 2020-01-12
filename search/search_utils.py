from search.searcher.Search import Search
from search.search_tests.search_test_functions import *

from os import path, getcwd
from json import load

# TODO:
#   Logging
#   Timing script
#   Read in multiple CSV index files for 1 run
#
# Flow:
#   Read config file
#   Read index into memory
#   Get pairwise euclidean distance for every feature
#   Sort euclidean distances
#   Calculate ranking

# Parse search config JSON file. Config structure:
#
# {
#     "input_path_base": "path/to/input/index/",
#     "input_name": "index_name/",
#     "output_path_base": "path/to/output/result/",
#     "output_name": "name_of_result_file.txt",
#     "metric": "name_of_similarity_metric",
#     "limit": "what does this do again"
# }
def search_json_read():
    config_path = path.join(getcwd(), 'search_params.json')
    assert(path.isfile(config_path)), "{} does not exist".format(config_path)
    print("Search config file path: ", config_path)

    with open(config_path) as f:
        config_file = load(f)
        input_path = config_file['input_path_base']
        input_name = config_file['input_name']
        output_path = config_file['output_path_base']
        output_name = config_file['output_name']
        metric = config_file['metric']

    config_dict = dict()
    config_dict['input'] = input_path + input_name
    config_dict['output'] = output_path + output_name
    config_dict['similarity_metric'] = metric

    return config_dict


def search_driver_run():
    # Read input JSON
    config = search_json_read()
    test_config_dict(config)

    # Instantiate search object
    search_object = Search(search_config=config)
    # test_search_object(search_object)

    # Create ranked lists and metrics
    search_object.results_create()
