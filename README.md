# Projet PFE SONDRA : SAR

# Getting Started

## Requirements

Before running any code, make sure to install all the dependancies by running 
```
pip3 install -r requirements.txt
```

## Data Preparation

### Download data
You can already download some data by running the download-data.sh script.
```
cd data
sudo chmod +x download-data.sh
./download-data.sh
```

### Build a training dataset 

To build a dataset that will be used for training we need to split the raw SAR data into several subimages. 
The idea is to crop squares of size 'IMAGE_SIZE' (that you can specify in the config.yaml file) from the SAR image. 
For example, if the SAR image is 10000x60000 pixels and IMAGE_SIZE is 1000, then we will generate 600 subimages of size 1000x1000 pixels. 

This can be done running the following commands
```
cd data
python3 dataset_builder.py --path_to_config ../config.yaml
```

The generated data will be stored in the data_files/train folder. 
The directory structure of your data folder should look like this
```
.
└── data_files
|   ├── raw_data
|   └── train
|       ├── low_resolution
|       └── high_resolution
├── data_reader.py
├── dataset_builder.py
├── download-data.sh
└── SARdataset.py
```

## Config

A config file is provided to store any 'global variable' that we would want to tune later on. 
The goal is to have all the variables centralized to easely change parameters when retraining a model for example and to have a clean code.

## Model Training


# Les participants

- côté SONDRA : 
    - Chengfang Ren <chengfang.ren@centralesupelec.fr>, 
    - hinostroza Israel <Israel.Hinostroza@centralesupelec.fr>
- côté étudiants : 
    - Youssef Adarrab <youssef.adarrab@student-cs.fr>, 
    - Daniel Colombo <daniel.colombo@student-cs.fr>, 
    - Alexandre Daly <alexandre.daly@student-cs.fr>
- encadrant CentraleSupélec : 
    - Jeremy Fix jeremy.fix@centralesupelec.fr>
