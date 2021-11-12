##############################################################################
# Some data reading functions
# Authored by Ammar Mian, 09/11/2018
# Completed by Chengfang Ren and Israel Hinostroza 10/11/2021 
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import numpy as np
import os, fnmatch
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

from skimage import data, img_as_float
from skimage import exposure


from matplotlib.colors import Normalize

matplotlib.rcParams['font.size'] = 8
path = r"/home/cfren/CEI_SONDRA/2021-2022/data/SSurge_15305_01/"

os.chdir(path)

# doc = r"SanAnd_26524_19002_011_190221_L090HH_04_BC_s1_1x1.slc"


class Uavsar_slc_stack_1x1():
    """A class to store data corresponding to a SLC stack of UAVSAR data
        * path = path to a folder containing the files obtained from UAVSAR (.slc, .ann, .llh)"""
    def __init__(self, path):
        self.path = path
        self.meta_data = {}
        self.llh_grid = {}
        self.slc_data = {}


    def read_data(self, polarisation=['HH', 'HV', 'VV'], segment=1, crop=None, singlefile = True):
        """ A method to read UAVSAR SLC 1x1 data stack
            Inputs:
                * polarisation = a list of polarisations to read
                * crop = if we want to read a portion of the image, a list of the form
                    [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1]"""

        # Obtain list of all files in directory
        listOfFiles = os.listdir(self.path)
        
        # Iterate on those files to search for an annotation file
        for entry in listOfFiles:  
        
            if fnmatch.fnmatch(entry, "*.ann"):

                    # Read the ann file to obtain metadata
                    self.meta_data[entry.split('.')[0]] = {} # initialise dict for the currect file
                    with open(self.path + entry, 'r') as f:
                        for line in f: # Iterate on each line
                            # Discard commented lines
                            line = line.strip().split(';')[0]
                            if not (line == ''):
                                category = ' '.join(line.split()[:line.split().index('=')-1])
                                value = ' '.join(line.split()[line.split().index('=')+1:])
                                self.meta_data[entry.split('.')[0]][category] = value


        # Read slc file corresponding to the segment of interest and crop it

        # First, we obtain a list containing the different filenames for each date
        # We put POL at the place of the polarisation and SEGMENT at the place of the segment in order to replace it after
        self.unique_identifiers_time_list = []
        for entry in list(self.meta_data.keys()):
            unique_identifiers_time = '_'.join(entry.split('_')[:-2])[:-2] + "POL_" + \
                                      '_'.join(entry.split('_')[-2:]) + '_sSEGMENT'
            if unique_identifiers_time not in self.unique_identifiers_time_list:
                self.unique_identifiers_time_list.append(unique_identifiers_time)
            
        # Then we read the files one by one for each polarisation and time
        if crop is not None:
            self.data = np.zeros((crop[1]-crop[0], crop[3]-crop[2], len(polarisation), 
                        len(self.unique_identifiers_time_list)), dtype='complex64')
            
            for t, entry_time in enumerate(self.unique_identifiers_time_list):
                for i_pol, pol in enumerate(polarisation):
                    # Read slc file at the given crop indexes
                    file_name = entry_time.replace('POL', pol).replace('SEGMENT', str(segment))
                    shape = (int(self.meta_data['_'.join(file_name.split('_')[:-1])]['slc_1_1x1 Rows']),
                             int(self.meta_data['_'.join(file_name.split('_')[:-1])]['slc_1_1x1 Columns']))   
                    print("Reading %s" % (self.path+file_name))
                    with open(self.path + file_name + '_1x1.slc', 'rb') as f:
                        f.seek((crop[0]*shape[1]+crop[2])*8, os.SEEK_SET)
                        for row in range(crop[1]-crop[0]):
                            self.data[row, :, i_pol,t] = np.fromfile(f, dtype=np.complex64, count=crop[3]-crop[2])
                            f.seek(((crop[0]+row)*shape[1]+crop[2])*8, os.SEEK_SET)
        else:
            for t, entry_time in enumerate(self.unique_identifiers_time_list):
                for i_pol, pol in enumerate(polarisation):
                    # Read whole slc file
                    file_name = entry_time.replace('POL', pol).replace('SEGMENT', str(segment))
                    print("Reading %s" % (self.path+file_name))
                    shape = (int(self.meta_data['_'.join(file_name.split('_')[:-1])]['slc_1_1x1 Rows']),
                             int(self.meta_data['_'.join(file_name.split('_')[:-1])]['slc_1_1x1 Columns']))               
                    temp_array = np.fromfile( self.path + file_name + '_1x1.slc', dtype='complex64').reshape(shape)
                    if t==0 and i_pol==0:
                        self.data = temp_array
                    else:
                        self.data = np.stack( (self.data, temp_array) )
                        
    def plot_equalized_img(self, data, method="equal", crop=None, bins = 256):
        """ A method to plot single UAVSAR SLC 1x1 data
            Inputs:
                * crop = [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1], a list of int, to read a portion of the image, it reads the whole image if None. 
                * method = "stretch" or "equal", based from histogram equalization, see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
                    
        """
        
        if crop is not None:
        
            img = np.log10(np.abs(data[crop[0]:crop[1], crop[2]:crop[3]]) + 1e-8 )
            
        else:
            
            img = np.log10(np.abs(data) + 1e-8)
        
        img = (img - img.min())/(img.max() - img.min()) #rescale between 0 and 1
            
        
        if method == "stretch":
            p2, p98 = np.percentile(img, (2, 98))
            img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
            
        elif method == "equal":
            img_rescale = exposure.equalize_hist(img)
            
        else:
            raise NameError("wrong 'method' or not defined")
        
        fig = plt.figure(figsize=(8, 8))
        image = img_as_float(img_rescale)
        plt.imshow(image, cmap=plt.cm.gray, aspect = img.shape[1]/img.shape[0])
        plt.show()

    def plot_mlpls_img(self, method="equal", all=False, crop=None, bins = 256):
        """ A method to plot multiples UAVSAR SLC 1x1 data
            Inputs:
                * crop = [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1], a list of int, to read a portion of the image, it reads the whole image if None. 
                * method = "stretch" or "equal", based from histogram equalization, see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html"""
            
        if all and len(self.data.shape)>2:
            for ii in range(self.data.shape[0]):
                self.plot_equalized_img(self.data[ii,:,:], method, crop, bins)
        elif len(self.data.shape)>2:
            self.plot_equalized_img(self.data[0,:,:], method, crop, bins)
        else:
            self.plot_equalized_img(self.data, method, crop, bins)
            
                

# Load an example image

sardata = Uavsar_slc_stack_1x1(path)
sardata.read_data(polarisation=['HH'])
sardata.plot_mlpls_img(crop=[20000,30000,2000,8000])

