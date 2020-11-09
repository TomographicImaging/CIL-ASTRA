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
from cil.framework import ImageGeometry, AcquisitionGeometry
from cil.plugins.astra.operators import AstraProjectorSimple, AstraProjector3DSimple, AstraProjectorFlexible
from cil.plugins.astra.operators import ProjectionOperator
import numpy as np
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

        N = 128
        angles = np.linspace(0, np.pi, 180, dtype='float32')

        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])
        
        ig = ag.get_ImageGeometry()

        
        ag3 = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['vertical', 'angle', 'horizontal'])

        ig3 = ag3.get_ImageGeometry()

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
        N = 128
        angles = np.linspace(0, np.pi, 180, dtype='float32')

        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])
        
        ig = ag.get_ImageGeometry()

        
        ag3 = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['vertical', 'angle', 'horizontal'])

        ig3 = ag3.get_ImageGeometry()

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

class TestProjectionOperator(unittest.TestCase):
    def setUp(self): 
        # Define image geometry.
        N = 128
        angles = np.linspace(0, np.pi, 180, dtype='float32')

        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])
        
        ig = ag.get_ImageGeometry()

        
        ag3 = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['vertical', 'angle', 'horizontal'])

        ig3 = ag3.get_ImageGeometry()

  
        ag_channel = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['channel', 'angle', 'horizontal'])\
                                .set_channels(2)

        ig_channel = ag_channel.get_ImageGeometry()

        
        ag3_channel = AcquisitionGeometry.create_Parallel3D()\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel((N, N), (0.1, 0.1))\
                                .set_labels(['channel','vertical', 'angle', 'horizontal'])\
                                .set_channels(2)

        ig3_channel = ag3_channel.get_ImageGeometry()

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
        A = ProjectionOperator(self.ig, self.ag, device='cpu')
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A = ProjectionOperator(self.ig_channel, self.ag_channel, device='cpu')
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        with self.assertRaises(NotImplementedError):
            A = ProjectionOperator(self.ig3, self.ag3, device='cpu')

    @unittest.skipIf(not use_cuda, "Astra not built with CUDA")
    def test_gpu(self):
        A = ProjectionOperator(self.ig, self.ag)
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A = ProjectionOperator(self.ig_channel, self.ag_channel)
        n = A.norm()
        print ("norm A GPU", n)
        self.assertAlmostEqual(n, self.norm, places=2)

        A3 = ProjectionOperator(self.ig3, self.ag3)
        n = A3.norm()
        print ("norm A3", n)
        self.assertAlmostEqual(n, self.norm, places=2)    

        A3_channel = ProjectionOperator(self.ig3_channel, self.ag3_channel)
        n = A3_channel.norm()
        print ("norm A4", n)
        self.assertAlmostEqual(n, self.norm, places=2)  
          