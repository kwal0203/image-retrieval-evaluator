# image-retrieval-system
Content based image retrieval system used for my undergraduate thesis in computer vision at [UNSW](https://www.engineering.unsw.edu.au/computer-science-engineering/). This tool was used to assess the effectiveness of image features for satellite image retrieval.


## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.


### Installation

Make sure you have Python version 3.X and following the steps below to get up and running.

To download the repository onto your system:

```
git clone https://github.com/kwal0203/image-retrieval-system.git
```

Move into the new directory:
```
cd image-retrieval-system
```

Create a virtual environment:
```
python3 -m venv env
```

Activate the virtual environment:
```
source env/bin/activate
```

Install dependencies:
```
pip3 install -r requirements.txt
```


### Usage

```
Basic workflow diagram
1. Train a model (or use pre-trained PyTorch model)
2. Configure feature extraction
3. Configure feature evaluation
4. Run feature extraction and/or feature evaluation
```

**Train a model**

Standard convolutional neural network based models (from the Pytorch model library) can be used out of the box in this program or user defined models may be provided. If not using a standard pre-trained model, the first step to using the program is training your specific model. A path to the state dictionary of this model will be required in the configuration step.

**Feature extraction configuration**

This program extracts features from images by inputting images into a neural network and capturing the activation values at a given layer. The feature extraction step works according to a configuration file which must be created in the program root directory named '''feature_params.json'''. The structure of the configuration file needs to be as follows:

```
{
    "input_path_base": "/path/to/parent/directory/of/image/folder/",
    "input_name": "/name/of/image/folder/",
    "output_path_base": "/path/to/directory/to/store/created/image/index/",
    "output_name": "name_of_image_index.csv",
    "feature_path": "/path/to/pytorch/model/", (leave blank if using pre-trained PyTorch model)
    "feature_name": "name_of_feature",
    "layer": integer layer representing layer of network to get feature from
}
```
input_path_base:  Path to the parent directory of the directory where the images are stored

input_name:       Name of the directory that contains the images

output_path_base: Path of the directory where you wish to store the create image index

output_name:      Name of the image index (the index is created as a .csv file)

feature_path:     Path to a PyTorch state dictionary for model you have trained (optional)

layer:            The layer of the network you wish to obtain the image feature from.


**Feature evaluation configuration**

The image index create in the previous step can be used to evaluate the performance of image features for the image retrieval task. Similar to above, a configuration file must be created in the program root directory named '''search_params.json'''. The structure of the configuration file needs to be as follows:

```
{
    "input_path_base": "/path/to/parent/directory/of/image/index/",
    "input_name": "name_of_image_index.csv",
    "output_path_base": "/path/to/directory/to/store/image_retrieval/results/",
    "output_name": "name_of_image_retrieval_results_file.txt",
    "metric": "name of similarity metric"
}
```
input_path_base:  Path to the directory containing the image index

input_name:       Name of image index file

output_path_base: Path to directory where you wish to store the image retrieval result file

output_name:      Name of file containing image retrieval results

metric:           Name of the similarity metric you wish to use.

**Run program**

Once the configuration files are setup and you are in the program root directory, run the program using the following command:

```
python3 main.py
```

## Authors

* **Kane Walter** - [LinkedIn](https://www.linkedin.com/in/kanewalter/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)
file for details.
