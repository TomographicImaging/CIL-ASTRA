#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:27:22 2019

@author: vaggelis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:32:54 2019

@author: evangelos
"""
from ccpi.framework import ImageData, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra
from ccpi.framework import ImageGeometry, AcquisitionGeometry
                            
import os       
import tomophantom
from tomophantom import TomoP2D
from ccpi.astra.operators import AstraProjectorSimple 

import numpy as np
import matplotlib.pyplot as plt

from ccpi.astra.processors import FBP

# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 256 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

#vox = [0.03, 0.1, 3, 10]
#vox = [6]

distance_source_origin = 500 # mm
distance_origin_detector = 250 # mm
detector_pixel_size = 0.8 # mm

# Compute magnification factor
mag = (distance_source_origin + distance_origin_detector) / distance_source_origin

# Real voxel size
voxel_real = detector_pixel_size / mag
angles_num = 180 # angles number
angles_rad = np.linspace(-np.pi, np.pi, angles_num) #angles*(np.pi/180.0)


vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('fanflat', mag, N, angles_rad, distance_source_origin, 0);
# fanflat with Astra
rec_id = astra.data2d.create( '-vol', vol_geom)
proj_id = astra.create_projector('line_fanflat',proj_geom,vol_geom)

sinogram_id, sino_cpu = astra.create_sino(phantom_2D, proj_id)
astra.data2d.delete(sinogram_id)
astra.data2d.delete(proj_id)
astra.data2d.delete(rec_id)

rec_id = astra.data2d.create( '-vol', vol_geom)
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

sinogram_id, sino_gpu = astra.create_sino(phantom_2D, proj_id)
astra.data2d.delete(sinogram_id)
astra.data2d.delete(proj_id)
astra.data2d.delete(rec_id)

plt.imshow(sino_cpu)
plt.colorbar()
plt.show()

plt.imshow(sino_gpu)
plt.colorbar()
plt.show()

plt.imshow(np.abs(sino_cpu - sino_gpu))
plt.colorbar()
plt.show()
#%%