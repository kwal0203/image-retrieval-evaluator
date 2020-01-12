from abc import ABCMeta, abstractmethod
from os import listdir, path
from skimage import io


class FeatureBase(object, metaclass=ABCMeta):
    def __init__(self, config):
        self.feature_name = config['feature_name']
        self.feature_path = config['feature_path']
        self.input_path = config['input']
        self.output_path = config['output']

    @abstractmethod
    def get_feature(self, image):
        pass

    # Iterate through all images and write feature to CSV for each one
    def index_create(self, feature):
        output_path = self.output_path
        input_path = self.input_path
        # assert(not path.isfile(output_path)), "{} exists".format(output_path)

        with open(output_path, "w+") as f:
            cnt = 0

            for filename in listdir(input_path):
                # Filename without file extension used as name of feature in CSV
                img_id = filename.split('.')[0]

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
