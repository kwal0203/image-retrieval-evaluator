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
pip3 install -m requirements.txt
```


### Usage

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
