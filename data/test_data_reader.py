import os

from data_reader import Uavsar_slc_stack_1x1


path = os.path.join(os.path.abspath(os.getcwd()), 'data_files', 'labels')
print(path)
os.chdir(path)

# Load an example image

sardata = Uavsar_slc_stack_1x1(path)
sardata.read_meta_data(polarisation=['HH'])

sardata.read_data(list(sardata.meta_data.keys())[0], crop = [0, 2000, 0, 2000])


sardata.plot_mlpls_img(savefig=True)


#plot originial SAR image 
sardata.plot_mlpls_img(savefig=True)

#SAR image halve downscaled in dual band with zero padding 
sardata.subband_process(list(sardata.slc_data.keys())[0], decimation = False)
sardata.plot_equalized_img(sardata.subimages[list(sardata.subimages.keys())[0]],name= "Original scale zero padding")

# #SAR image halve downscaled in dual band
sardata.subband_process(list(sardata.slc_data.keys())[0], decimation = True)
sardata.plot_equalized_img(sardata.subimages[list(sardata.subimages.keys())[0]],name= "downscale")
# 
# #SAR image halve downscaled in dual band with hanning smoothing
sardata.subband_process(list(sardata.slc_data.keys())[0], decimation = True, wd="hanning")
sardata.plot_equalized_img(sardata.subimages[list(sardata.subimages.keys())[0]],name= "downscale with hanning smoothing")

