import os
import numpy as np

import torch
from torch.utils.data import Dataset
from skimage import exposure

class SARdataset(Dataset):
    """Store the SAR data into a torch dataset like object. 

    Args:
        Dataset (class): pytorch dataset object 
    """

    def __init__(self, root):
        """
        Args:
            root (str): absolute path of the data files 
            image_type (str): type of bandwith used for SAR images. Defaults to 'HH'.
            preprocessing (str): type of preprocessing to perform. Can be either 'padding', 'hanning' or None. Defaults to 'None'.
        """

        self.root = root
        self.files_names = [f for f in os.listdir(os.path.join(self.root,'high_resolution')) if 
                            os.path.isfile(os.path.join(self.root,'high_resolution', f))]
        print(self.files_names)


    def __getitem__(self, idx):
        """Retrieve the i-th item of the dataset

        Args:
            idx (int): idx-th item to retrieve

        Returns:
            image_input, image_target: the low resolution image and the high resolution image
        """

        image_input = np.load(os.path.join(self.root,'high_resolution',self.files_names[idx]))
        image_target = np.load(os.path.join(self.root,'low_resolution',self.files_names[idx]))

        return apply_processing(image_input), apply_processing(image_target)

    def __len__(self):
        """Operator len that returns the size of the dataset 

        Returns:
            int: length of the dataset
        """
        return len(self.files_names)


def apply_processing(data,method='equal'):
    """A function to have the images in log mode

    Args:
        data (np.array): the images as a numpy array 
    
    Return: 
        img (np.array): the processed image
    """
             
    img = np.log10(np.abs(data) + 1e-8)
    img = (img - img.min())/(img.max() - img.min()) #rescale between 0 and 1
            
    if method == "stretch":
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        
    elif method == "equal":
        img_rescale = exposure.equalize_hist(img)
        
    else:
        raise NameError("wrong 'method' or not defined")
            
    return img_rescale


dataset = SARdataset("./data_files/train")
print(dataset[35])

