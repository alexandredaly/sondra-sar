import os
import numpy as np

import torch
from torch.utils.data import Dataset

from data_reader import Uavsar_slc_stack_1x1


class SARdataset(Dataset):
    """Store the SAR data into a torch dataset like object. 

    Args:
        Dataset (class): pytorch dataset object 
    """

    def __init__(self, root, image_type ='HH', preprocessing = None):
        """
        Args:
            root (str): absolute path of the data files 
            image_type (str): type of bandwith used for SAR images. Defaults to 'HH'.
            preprocessing (str): type of preprocessing to perform. Can be either 'padding', 'hanning' or None. Defaults to 'None'.
        """

        self.root = root
        self.files_names = [f for f in os.listdir(self.root) if 
                            os.path.isfile(os.path.join(self.root, f)) and f.endswith(".slc")]
        self.preprocessing = preprocessing
        self.data_reader = Uavsar_slc_stack_1x1(self.root)
        sardata.read_meta_data(polarisation=[image_type])
        sardata.read_data(list(sardata.meta_data.keys())[0] crop = [200, 800, 200, 800])


    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): item index
        """
        
        # Coresponding identifier of the index
        id = self.files_names[idx]

        # Retrieve the target image (high resolution image)
        target = self.data_reader.slc_data[id]
        
        # Process the image to decrease the resolution
        if self.preprocessing == 'padding':
            self.data_reader.subband_process(list(sardata.slc_data.keys())[idx], decimation = True)

        elif self.preprocessing == 'hanning':
            self.data_reader.subband_process(list(sardata.slc_data.keys())[idx], decimation = True, wd="hanning")

        else:
            self.data_reader.subband_process(list(sardata.slc_data.keys())[idx], decimation = False)

        image = self.data_reader.subimages[id]

        return image, target

    def __len__(self):
        return len(list(self.data_reader.slc_data.keys()))