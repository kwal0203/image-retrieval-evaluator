from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import csv


class Search:
    def __init__(self, limit, metric):
        self.limit = limit
        self.metric = metric
        self.features = None
        self.result = []
        self.names = []

    # Return similarity between two image features FOR USER MODE
    # def get_similarity(self):
    #     return self.limit

    # Print results to screen/file
    def print_results(self):
        _map = round(self.result[0], 2)
        p_5 = round(self.result[1], 2)
        p_10 = round(self.result[2], 2)
        p_25 = round(self.result[3], 2)
        p_50 = round(self.result[4], 2)
        p_100 = round(self.result[5], 2)
        print("p_5:   ", p_5)
        print("p_10:  ", p_10)
        print("p_25:  ", p_25)
        print("p_50:  ", p_50)
        print("p_100: ", p_100)
        # TODO: Find logging library compatible with Python 3.6
        # logging.info("----- RESULTS -----")
        # logging.info("{} {} {} {} {} {}".format(
        #         _map, p_5, p_10, p_25, p_50, p_100))

    # A distance matrix is computed from the whole set of image features. Each
    # column is sorted and various metrics calculated on the resulting list.
    def results_create(self):
        _row_names = self.names

        dist = euclidean_distances(self.features)
        print("Distance matrix calculation finished")

        distance_df = pd.DataFrame(dist, index=_row_names, columns=_row_names)
        print("Data-frame construction finished")

        _map = 0
        map_count = 0

        # Dictionary to store precision at k values as they are calculated.
        # All values initialized to 0 to begin with.
        # k_list = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
        k_list = [5, 10, 25, 50, 100, 250, 500]
        prec_arr = {i: 0 for i in k_list}

        for col_idx, col_name in enumerate(distance_df.columns):
            feature = distance_df.loc[col_name]
            image_class = ''.join([x for x in col_name if not x.isdigit()])

            # Sort retrieval results and discard the first item (this is the
            # query feature itself). We then map the sorted results to 0's and
            # 1's depending on if the given item is in the same class as the
            # query.
            sorted_df = feature.sort_values()[1:]
            preds = [1 if image_class in j else 0 for j in sorted_df.index]

            # print(sorted_df.shape)
            # print(sorted_df[:10])
            # print(preds[:10])
            # print("Image name:  ", col_name)
            # print("Image class: ", image_class)
            # sys.exit()

            # Precision list
            prec_at_k_list = []
            hit = 0

            # Store intermediate calculations for MAP and P@K
            for idx, pred in enumerate(preds):
                _idx = idx + 1

                # For MAP calculation
                if pred == 1:
                    hit += 1
                    prec_at_k_list.append(hit / _idx)

                # For P@K calculations
                if _idx in prec_arr:
                    prec_arr[_idx] += (hit / _idx)

            if len(prec_at_k_list) > 0:
                average_precision = np.mean(prec_at_k_list)
            else:
                average_precision = 0

            _map += average_precision
            map_count += 1

            if map_count % 250 == 0:
                print("[INFO] - Query number: {}".format(map_count))

        _map /= map_count
        _map *= 100

        # Store final results in a list then call print function
        self.result.append(_map)
        for k in k_list:
            self.result.append(100 * (prec_arr[k] / self.limit))

        self.print_results()

    # Create list of features in numpy array format. Do this so we don't
    # have to read the CSV file for each query.
    def read_index(self, idx_path, num_files, feature_size):
        with open(idx_path) as f:
            self.features = np.zeros(shape=(num_files, feature_size))

            print("Num files: ", num_files)

            idx_count = 0
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                # print("IDX:       ", idx)
                # print("Row:       ", row[0:25])
                # print("Row.shape: ", len(row[1:]))
                # sys.exit()
                self.features[idx] = np.array(row[1:])
                self.names.append(row[0])
                # print("name: {}, IDX: {}".format(row[0], idx_count))
                idx_count += 1
                # if idx_count % 250 == 0:
                if idx_count % 100 == 0:
                    # print("[INFO] - Reading row {}".format(idx_count))
                    print("[INFO] - Reading row {}".format(idx_count))

            # print("Names[0]:      ", names[0])
            # print("len(Names): ", len(names))
            # print("Index: ", image_index[0][0:25])
            # print("Names[2099]:      ", names[2099])
            # print("len(Names): ", len(names))
            # print("Index: ", image_index[2099][0:25])
            # print("Index count:        ", idx_count)
            # print("len(names):         ", len(names))
            # print("image_index.shape:  ", image_index.shape)
            # sys.exit()
