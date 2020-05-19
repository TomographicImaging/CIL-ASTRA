# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import unittest
import numpy as np
from ccpi.framework import ImageGeometry, AcquisitionGeometry
from ccpi.framework import ImageData, AcquisitionData
from ccpi.framework import BlockDataContainer, TestData
import functools

from ccpi.optimisation.operators import Gradient, Identity, BlockOperator
from ccpi.optimisation.operators import LinearOperator

from ccpi.astra.operators import AstraProjectorSimple, AstraProjector3DSimple


class TestAstraSimple(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        N = 256

        loader = TestData()
        ph = loader.load(TestData.SIMPLE_PHANTOM_2D,size=(N,N))

        ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1)
        im_data = ig.allocate()
        im_data.fill(ph.as_array())

        # show(im_data, title = 'TomoPhantom', cmap = 'inferno')
        # Create AcquisitionGeometry and AcquisitionData 
        detectors = N
        angles = np.linspace(0, np.pi, 180, dtype='float32')
        ag = AcquisitionGeometry('parallel','2D', angles, detectors,
                                pixel_size_h = 0.1)
        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        device = 'gpu'
        A = AstraProjectorSimple(ig, ag, device = device)

        data = A.direct(im_data)

        ig3 = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        voxel_size_z = 0.1)
        # im_data3 = ig3.allocate()

        
        ag3 = AcquisitionGeometry('parallel','3D', angles, detectors,
                                pixel_size_h = 0.1,
                                pixel_size_v = 0.1)
        A3 = AstraProjector3DSimple(ig3, ag3)
        self.im_data = im_data
        self.data = data
        self.A = A
        self.ig = ig
        self.ag = ag
        self.A3 = A3
        self.ig3 = ig3
        self.ag3 = ag3

    def test_norm_simple2D(self):
        # test exists
    
        n = self.A.norm()
        print ("norm A", n)
        self.assertTrue(True)
        
    def skip_test_norm_simple3D(self):
        # test exists
        n = self.A3.norm()
        print ("norm A3", n)
        self.assertTrue(True)
    