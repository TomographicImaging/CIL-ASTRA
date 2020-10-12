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

from ccpi.astra.operators import AstraProjectorSimple, AstraProjector3DSimple, AstraProjectorFlexible
from ccpi.astra.operators import AstraOperator

import astra
use_cuda = True
try:
    astra.test_CUDA()
except RuntimeError as re:
    print (re)
    use_cuda = False
except:
    use_cuda = False



class TestAstraSimple(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        N = 128

        ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1)
        
        detectors = N
        angles = np.linspace(0, np.pi, 180, dtype='float32')
        ag = AcquisitionGeometry(geom_type='parallel',
                                 dimension='2D', 
                                 angles=angles, 
                                 pixel_num_h=detectors,
                                 pixel_size_h = 0.1,
                                 dimension_labels=['angle','horizontal'],
                                 angle_unit = 'radian')
        
        ig3 = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        voxel_size_z = 0.1)
        
        
        ag3 = AcquisitionGeometry(geom_type = 'parallel',
                                 dimension= '3D', 
                                 angles=angles, 
                                 pixel_num_h = detectors,
                                 pixel_num_v = detectors,
                                 pixel_size_h = 0.1,
                                 pixel_size_v = 0.1,
                                 dimension_labels=['vertical','angle','horizontal'],
                                 angle_unit = 'radian')
        self.ig = ig
        self.ag = ag
        self.ig3 = ig3
        self.ag3 = ag3
        self.norm = 14.85

    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_norm_simple2D_gpu(self):
        # test exists
        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        device = 'gpu'
        A = AstraProjectorSimple(self.ig, self.ag, device = device)

        n = A.norm()
        print ("norm A GPU", n)
        self.assertTrue(True)
        self.assertAlmostEqual(n, self.norm, places=2)

    def test_norm_simple2D_cpu(self):
        # test exists
        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        device = 'cpu'
        A = AstraProjectorSimple(self.ig, self.ag, device = device)

        n = A.norm()
        print ("norm A CPU", n)
        self.assertTrue(True)
        self.assertAlmostEqual(n, self.norm, places=2)
    
    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_norm_simple3D_gpu(self):
        # test exists
        A3 = AstraProjector3DSimple(self.ig3, self.ag3)
        n = A3.norm()
        print ("norm A3", n)
        self.assertTrue(True)
        self.assertAlmostEqual(n, self.norm, places=2)

class TestAstraFlexible(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        N = 128

        ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1)
        
        detectors = N
        angles = np.linspace(0, np.pi, 180, dtype='float32')
        ag = AcquisitionGeometry(geom_type='parallel',
                                 dimension='2D', 
                                 angles=angles, 
                                 pixel_num_h=detectors,
                                 pixel_size_h = 0.1,
                                 dimension_labels=['angle','horizontal'],
                                 angle_unit = 'radian')
        
        ig3 = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        voxel_size_z = 0.1)
        
        
        ag3 = AcquisitionGeometry(geom_type = 'parallel',
                                 dimension= '3D', 
                                 angles=angles, 
                                 pixel_num_h = detectors,
                                 pixel_num_v = detectors,
                                 pixel_size_h = 0.1,
                                 pixel_size_v = 0.1,
                                 dimension_labels=['vertical','angle','horizontal'],
                                 angle_unit = 'radian')
        self.ig = ig
        self.ag = ag
        self.ig3 = ig3
        self.ag3 = ag3
        self.norm = 14.85

    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_norm_flexible2D_gpu(self):
        # test exists
        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        A = AstraProjectorFlexible(self.ig, self.ag)
        n = A.norm()
        print ("norm A GPU", n)
        self.assertTrue(True)
        self.assertAlmostEqual(n, self.norm, places=2)
    
        ag_2 = self.ag.copy()
        ag_2.dimension_labels = ['horizontal','angle']
        with self.assertRaises(ValueError):
            A = AstraProjectorFlexible(self.ig, ag_2)

        ig_2 = self.ig3.copy()
        ig_2.dimension_labels = ['horizontal_x','horizontal_y']
        with self.assertRaises(ValueError):
            A = AstraProjectorFlexible(ig_2, self.ag)

    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_norm_flexible3D_gpu(self):
        # test exists
        A3 = AstraProjectorFlexible(self.ig3, self.ag3)
        n = A3.norm()
        print ("norm A3", n)
        self.assertTrue(True)
        self.assertAlmostEqual(n, self.norm, places=2)    

        ag3_2 = self.ag3.copy()
        ag3_2.dimension_labels = ['angle','vertical','horizontal']
        with self.assertRaises(ValueError):
            A3 = AstraProjectorFlexible(self.ig3, ag3_2)

        ig3_2 = self.ig3.copy()
        ig3_2.dimension_labels = ['horizontal_y','vertical','horizontal_x']
        with self.assertRaises(ValueError):
            A3 = AstraProjectorFlexible(ig3_2, self.ag3)

class TestAstraOperator(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        N = 128

        ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1)
        
        detectors = N
        angles = np.linspace(0, np.pi, 180, dtype='float32')
        ag = AcquisitionGeometry(geom_type='parallel',
                                 dimension='2D', 
                                 angles=angles, 
                                 pixel_num_h=detectors,
                                 pixel_size_h = 0.1,
                                 dimension_labels=['angle','horizontal'],
                                 angle_unit = 'radian')

        ig_channel = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        channels=2)
        
        ag_channel = AcquisitionGeometry(geom_type='parallel',
                                 dimension='2D', 
                                 angles=angles, 
                                 pixel_num_h=detectors,
                                 pixel_size_h = 0.1,
                                 dimension_labels=['channel','angle','horizontal'],
                                 angle_unit = 'radian',
                                 channels=2)
                                         
        ig3 = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        voxel_size_z = 0.1)
        
        
        ag3 = AcquisitionGeometry(geom_type = 'parallel',
                                 dimension= '3D', 
                                 angles=angles, 
                                 pixel_num_h = detectors,
                                 pixel_num_v = detectors,
                                 pixel_size_h = 0.1,
                                 pixel_size_v = 0.1,
                                 dimension_labels=['vertical','angle','horizontal'],
                                 angle_unit = 'radian')

        ig3_channel = ImageGeometry(voxel_num_x = N, voxel_num_y = N, voxel_num_z=N, 
                        voxel_size_x = 0.1,
                        voxel_size_y = 0.1,
                        voxel_size_z = 0.1,
                        channels=2)
        
        ag3_channel = AcquisitionGeometry(geom_type = 'parallel',
                                 dimension= '3D', 
                                 angles=angles, 
                                 pixel_num_h = detectors,
                                 pixel_num_v = detectors,
                                 pixel_size_h = 0.1,
                                 pixel_size_v = 0.1,
                                 dimension_labels=['channel','vertical','angle','horizontal'],
                                 angle_unit = 'radian',
                                 channels= 2)

        self.ig = ig
        self.ag = ag
        self.ig_channel = ig_channel
        self.ag_channel = ag_channel       
        self.ig3 = ig3
        self.ag3 = ag3
        self.ig3_channel = ig3_channel
        self.ag3_channel = ag3_channel
        self.norm = 14.85

    def test_cpu(self):
        A = AstraOperator(self.ig, self.ag, device='cpu')
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A = AstraOperator(self.ig_channel, self.ag_channel, device='cpu')
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        with self.assertRaises(NotImplementedError):
            A = AstraOperator(self.ig3, self.ag3, device='cpu')

    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_gpu(self):
        A = AstraOperator(self.ig, self.ag)
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A = AstraOperator(self.ig_channel, self.ag_channel)
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A3 = AstraOperator(self.ig3, self.ag3)
        n = A3.norm()
        print ("norm A3", n)
        self.assertAlmostEqual(n, self.norm, places=2)    

        A3_channel = AstraOperator(self.ig3_channel, self.ag3_channel)
        n = A3_channel.norm()
        print ("norm A4", n)
        self.assertAlmostEqual(n, self.norm, places=2)  
          