from search.searcher.Search import Search
from search.search_tests.search_test_functions import *

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

# Parse config JSON file. Config structure:
#
# {
#     "input_path_base": "path/to/input/index/",
#     "input_name": "name_of_index.csv",
#     "output_path_base": "path/to/output/",
#     "output_name": "name_of_result_file.txt",
#     "feature_path": "path/to/model/", (leave blank for histogram features)
#     "feature_name": "name_of_feature",
#     "metric": "euclidean"
# }


def search_json_read():
    return 1


# Instantiate object for requested similarity metric
def search_object_create(limit, metric):
    return Search(limit=limit, metric=metric)


def search_driver_run():
    # Read input JSON
    config = search_json_read()
    test_config_dict(config)

    # Instantiate search object
    # search_object = search_object_create(CONFIG_LIMIT, CONFIG_METRIC)
    # test_search_object(search_object)

    # Create ranked lists and metrics
    # search_object.results_create()
