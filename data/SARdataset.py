import os
import numpy as np

import torch
from torch.utils.data import Dataset

from read_sar_data import Uavsar_slc_stack_1x1


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
            preprocessing (str): type of preprocessing to perform. Can be either 'zero padding', 'hanning smoothing' or None. Defaults to 'None'.
        """

        self.root = root
        self.files_names = [os.path.join(self.root, f) for f in os.listdir(self.root) if 
os.path.isfile(os.path.join(self.root, f))]
        self.preprocessing = preprocessing



    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): item index
        """

        if self.preprocessing == 'zero padding':

        elif self.preprocessing == 'hanning smoothing':
        
        else:
        
    def __len__(self):
        return len(self.files_names)





