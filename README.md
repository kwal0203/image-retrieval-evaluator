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

This program extracts features from images by inputting images into a neural network and capturing the activation values at a given layer. The feature extraction step works according to the configuration file '''feature_params.json'''. The structure of the configuration file is as follows:

```

```


This program inputs images into the specified model and extracts activation values from intermediate layers for use as an image feature.


```
1. Train a model

2. Put config into feature_params.json to create index

3. Put config into search_params.json to evaluate image retrieval performance

4. python main.py
```

## Authors

* **Kane Walter** - [LinkedIn](https://www.linkedin.com/in/kanewalter/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)
file for details.
