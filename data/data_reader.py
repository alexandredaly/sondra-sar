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

# Need to install scikit-image to use the following modules
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import window

from matplotlib.colors import Normalize


class Uavsar_slc_stack_1x1():
    """A class to store data corresponding to a SLC stack of UAVSAR data
        * path = path to a folder containing the files obtained from UAVSAR (.slc, .ann, .llh)
        * 
        """
    def __init__(self, path):
        self.path = path         # folder of SAR data
        self.meta_data = {}      # all images metadata
        # self.llh_grid = {}     # not used now
        self.slc_data = {}       # SAR images
        self.subband_header = {} # Characteristics for subband processing
        self.subimages = {}
        self.count = 0 
        
    
    def read_meta_data(self, polarisation=['HH', 'HV', 'VV']):
        """ A method to read UAVSAR SLC 1x1 meta data (*.ann file)
            Contain SAR metadata for all segments 
            Inputs:
                * polarisation = a list of polarisations to read
        """

        # Obtain list of all files in directory
        listOfFiles = os.listdir(self.path)
        print(listOfFiles)
        
        # Iterate on those files to search for an annotation file
        for entry in listOfFiles:  
        
            if fnmatch.fnmatch(entry, "*.ann"):
                
                unique_identifier = entry.split('.')[0]
                pol = unique_identifier.split('_')[-3][-2:] # Extract the polarisation of the current file 
                # Read metadata only if selected polarisation
                if pol in polarisation:
                    
                    if unique_identifier in list(self.meta_data.keys()):
                        raise NameError('file name already used')
                    
                    # Read the ann file to obtain metadata
                    self.meta_data[unique_identifier] = {} # initialise dict for the currect file
                
                    with open(os.path.join(self.path,entry), 'r') as f:
                        for line in f: # Iterate on each line
                            # Discard commented lines
                            line = line.strip().split(';')[0]
                            if not (line == ''):
                                category = ' '.join(line.split()[:line.split().index('=')-1])
                                value = ' '.join(line.split()[line.split().index('=')+1:])
                                self.meta_data[entry.split('.')[0]][category] = value
                    
 
    def read_subband_header(self, seg, crop, file_name, meta_identifier):
        
        # Read characteristics for subband processing
        
        self.subband_header[file_name] =  {}
        self.subband_header[file_name]['AzPixelSz'] = float(self.meta_data[meta_identifier]['1x1 SLC Azimuth Pixel Spacing'])
        self.subband_header[file_name]['RgPixelSz'] = float(self.meta_data[meta_identifier]['1x1 SLC Range Pixel Spacing'])
        self.subband_header[file_name]['SquintAngle'] = float(self.meta_data[meta_identifier]['Global Average Squint Angle'])
        self.subband_header[file_name]['cmWavelength'] = float(self.meta_data[meta_identifier]['Center Wavelength'])
        self.subband_header[file_name]['AzCnt'] = int(self.meta_data[meta_identifier]['slc_'+str(seg)+'_1x1 Rows'])
        self.subband_header[file_name]['RgCnt'] = int(self.meta_data[meta_identifier]['slc_'+str(seg)+'_1x1 Columns'])
        
        if crop is not None:
            self.subband_header[file_name]['Crop'] = crop
        else:
            self.subband_header[file_name]['Crop'] = None  
        
    
    def read_data(self, meta_identifier, segment=[1], crop=None):
        """ A method to read UAVSAR SLC 1x1 data stack
            Inputs:
                * polarisation = a list of polarisations to read
                * crop = if we want to read a portion of the image, a list of the form
                    [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis
                     1]
                * Be careful data should be stored in matrix order:  
                axis 0 -> azimuth & axis 1 -> range"""
                
        for seg in segment:
            
            if seg <= int(self.meta_data[meta_identifier]['Number of Segments']):
                file_name = meta_identifier + '_s'+ str(seg)+ '_1x1.slc'
                data_path = os.path.join(self.path,file_name)
                print(data_path)

                
                if os.path.isfile(data_path):
                    
                    if  file_name in list(self.slc_data.keys()):
                        
                        print("Warning file", file_name, "will be erased by the new read" )
                        
                    self.read_subband_header(seg, crop, file_name, meta_identifier)
                    
                    print("Reading %s" % (data_path))

                    shape = (self.subband_header[file_name]['AzCnt'], self.subband_header[file_name]['RgCnt'])
    
                    if crop is not None:
                        
                        temp_array = np.zeros((crop[1]-crop[0], crop[3]-crop[2]), dtype= np.complex64)    
                        
                        with open(data_path, 'rb') as f:
                            f.seek((crop[0]*shape[1]+crop[2])*8, os.SEEK_SET)
                            for row in range(crop[1]-crop[0]):
                                temp_array[row, :] = np.fromfile(f, dtype=np.complex64, count=crop[3]-crop[2])
                                f.seek(((crop[0]+row)*shape[1]+crop[2])*8, os.SEEK_SET)
                        
                    else:
                        
                        temp_array = np.fromfile( data_path, dtype=np.complex64).reshape(shape)
                        print(temp_array.shape)

                    self.slc_data[file_name] = temp_array
                    del temp_array

                        
    def plot_amp_img(self, cplx_image):
        plt.figure()
        plt.imshow(20*np.log10(np.abs(cplx_image)+1e-15), cmap=plt.cm.gray, aspect = cplx_image.shape[1]/cplx_image.shape[0])
        plt.colorbar()
        plt.show()
                        
    def plot_equalized_img(self, data, name, method="equal", crop=None, bins = 256, savefig = False):
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
        plt.title(name)
        plt.axis('equal')
        if savefig:
            plt.savefig(self.path + "stuff" +'.png')

        plt.show()
        
        
        

    def plot_mlpls_img(self, method="equal", all=False, crop=None, bins = 256, savefig=False):
        """ A method to plot multiples UAVSAR SLC 1x1 data
            Inputs:
                * crop = [lowerIndex axis 0, UpperIndex axis 0, lowerIndex axis 1, UpperIndex axis 1], a list of int, to read a portion of the image, it reads the whole image if None. 
                * method = "stretch" or "equal", based from histogram equalization, see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
                * all = True for plotting all data in self.data, False for plotting the 1st data
        """
        print(self.slc_data, 'slc')
        if bool(self.slc_data):
            if all:
                for data_name in list(self.slc_data.keys()):
                    data = self.slc_data[data_name]
                    print(self.slc_data[data_name])
                    self.plot_equalized_img(data, method, crop, bins, savefig)
            else:
                self.plot_equalized_img(self.slc_data[list(self.slc_data.keys())[0]],"Original image", method, crop, bins, savefig)
        else:
            raise KeyError("Empty dictionary")
            
    
    def subband_process(self, identifier, decimation = True, wd = None):
        """ A method to decompose the original image in the 2D spectral (dual range x dual azimuth) domain 
            Inputs:
                *identifier -> self.slc_data.keys() to select the data
                *decimation = True to halve pixels numbers for each dimension 
                            = False to keep the same but using zero padding method
                *wd         = choose window to mitigate secondary lobes see help(window) for windows option
                *
        """
        
        data = self.slc_data[identifier]
        RgPixelSz = self.subband_header[identifier]['RgPixelSz']
        AzPixelSz = self.subband_header[identifier]['AzPixelSz']
        RgCnt = self.subband_header[identifier]['RgCnt']
        AzCnt = self.subband_header[identifier]['AzCnt']
        crop = self.subband_header[identifier]['Crop']

        
        # Spatial coordinates of the SAR image
        
        if crop is not None:
            
            RgCnt = crop[3] - crop[2]  #Update the Azimuth & Range count if crop
            AzCnt = crop[1] - crop[0]
            
        SarRange = RgPixelSz * np.arange(-RgCnt/2, RgCnt/2) # radial axis (x axis / axis 1 in the image)
        SarAzimuth = AzPixelSz * np.arange(-AzCnt/2, AzCnt/2) # azimuth axis (y axis /axis 0 in the image)
        
        # spatial grid 
        
        (RRange, AAzimuth) = np.meshgrid(SarRange, SarAzimuth)
        
        # krange and kazimuth spectral dual variables
        
        kcentral = 2/self.subband_header[identifier]["cmWavelength"]*100 # k = 2/lambda with lambda in meter
        
        # Angle de viser au sol
        deport = (90 - self.subband_header[identifier]["SquintAngle"])*np.pi/180
        
        # kudop = fdop / vavion;
        # kudopcentral = kcentral * np.sin(deport)
        
        krange = kcentral*np.cos(deport) + (1/RgPixelSz)*np.arange(- 1/2, 1/2, 1/RgCnt)
        kazimuth = kcentral * np.sin(deport) + (1/AzPixelSz)*np.arange(- 1/2, 1/2, 1/AzCnt)
        

        # Add an offset in the dual space (SAR process) ~ (FFT shift)
        
        data = data * np.exp(-2*np.pi* 1j *(RRange * krange.min() + AAzimuth * kazimuth.min()), dtype= np.complex64) 
        
        spectre = np.fft.fft2(data)
        
        # Filtering in frequency -> sub band and Aperture -> theta
        # fcos(theta) = krange & fsin(theta) = kazimuth 
        
        # Spectral grid
        (KKrange, KKazimuth) = np.meshgrid(krange, kazimuth)
        
        frequence = np.sqrt(KKrange**2 + KKazimuth**2)
        theta = np.arctan2(KKazimuth, KKrange)
        
        f_min = frequence.min()
        f_max = frequence.max()
        theta_min = theta.min()
        theta_max = theta.max()
        
        sigma_f = (f_max - f_min) / 2;   #demie bande frequencielle
        sigma_t = (theta_max - theta_min) / 2; #demie bande angulaire
        f_centre = (f_max + f_min) / 2;  # frequence centrale de la bande
        theta_centre = (theta_max + theta_min) / 2 # angule central de la bande
        
        
        # Boolean filter (Centered)
        Filter = (np.abs(frequence - f_centre) <= sigma_f/3) * (abs(theta-theta_centre) <= sigma_t/3)
        
        sub_spectre = Filter * spectre
        
        # # # Checking spetral information
        # 
        # self.plot_amp_img(spectre)
        # self.plot_amp_img(sub_spectre)
        
        
        if decimation:
            #Décimation par 2 de chaque dim, crop central
            
            sub_spectre = sub_spectre[sub_spectre.shape[0]//4:(3*sub_spectre.shape[0])//4, sub_spectre.shape[1]//4:(3*sub_spectre.shape[1])//4] 
            # self.plot_amp_img(sub_spectre)

        
        if wd is not None:
            
            sub_spectre = window(wd, sub_spectre.shape ) * sub_spectre
        
        if  identifier in list(self.subimages.keys()):
                        
            print("Warning sub images", identifier, "will be erased by the processing" )
        
        self.subimages[identifier]  = np.fft.ifft2(sub_spectre)
        print(len(self.subimages.items()))



