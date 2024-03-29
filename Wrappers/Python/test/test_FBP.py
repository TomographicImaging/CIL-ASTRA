# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from cil.framework import AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP



import unittest
import numpy as np

try:
    import astra
    has_astra = True
except ImportError as re:
    print (re)
    has_astra = False

class TestProcessors(unittest.TestCase):
    def setUp(self): 
        #%% Setup Geometry
        voxel_num_xy = 255
        voxel_num_z = 15
        self.cs_ind = (voxel_num_z-1)//2


        src_to_obj = 500
        src_to_det = src_to_obj

        pix_size = 0.2
        det_pix_x = voxel_num_xy
        det_pix_y = voxel_num_z

        num_projections = 360
        angles = np.linspace(0, 2*np.pi, num=num_projections, endpoint=False)

        self.ag_cone = AcquisitionGeometry.create_Cone3D([0,-src_to_obj,0],[0,src_to_det-src_to_obj,0])\
                                     .set_angles(angles, angle_unit='radian')\
                                     .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                     .set_labels(['vertical','angle','horizontal'])

        self.ag_parallel = AcquisitionGeometry.create_Parallel3D()\
                                     .set_angles(angles, angle_unit='radian')\
                                     .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                     .set_labels(['vertical','angle','horizontal'])

        self.ig_3D = self.ag_parallel.get_ImageGeometry()

        #%% Create phantom
        kernel_size = voxel_num_xy
        kernel_radius = (kernel_size - 1) // 2
        y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]

        circle1 = [5,0,0] #r,x,y
        dist1 = ((x - circle1[1])**2 + (y - circle1[2])**2)**0.5

        circle2 = [5,100,0] #r,x,y
        dist2 = ((x - circle2[1])**2 + (y - circle2[2])**2)**0.5

        circle3 = [25,0,100] #r,x,y
        dist3 = ((x - circle3[1])**2 + (y - circle3[2])**2)**0.5

        mask1 =(dist1 - circle1[0]).clip(0,1) 
        mask2 =(dist2 - circle2[0]).clip(0,1) 
        mask3 =(dist3 - circle3[0]).clip(0,1) 
        phantom = 1 - np.logical_and(np.logical_and(mask1, mask2),mask3)

        self.golden_data = self.ig_3D.allocate(0)
        for i in range(4):
            self.golden_data.fill(array=phantom, vertical=7+i)

        self.golden_data_cs = self.golden_data.subset(vertical=self.cs_ind)


    @unittest.skipUnless(has_astra and astra.use_cuda(), "Astra not built with CUDA")
    def test_FBPgpu(self):

        #2D cone
        ag = self.ag_cone.subset(vertical='centre')
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='gpu')
        fp_2D = A.direct(self.golden_data_cs)

        fbp = FBP(ig, ag, 'gpu')
        fbp.set_input(fp_2D)
        fbp_2D_cone = fbp.get_output()
        np.testing.assert_allclose(fbp_2D_cone.as_array(),self.golden_data_cs.as_array(),atol=0.6)

        #3D cone
        ag = self.ag_cone
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='gpu')
        fp_3D = A.direct(self.golden_data)

        fbp = FBP(ig, ag, 'gpu')
        fbp.set_input(fp_3D)
        fbp_3D_cone = fbp.get_output()
        np.testing.assert_allclose(fbp_3D_cone.as_array(),self.golden_data.as_array(),atol=0.6)

        #2D parallel
        ag = self.ag_parallel.subset(vertical='centre')
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='gpu')
        fp_2D = A.direct(self.golden_data_cs)

        fbp = FBP(ig, ag, 'gpu')
        fbp.set_input(fp_2D)
        fbp_2D_parallel = fbp.get_output()
        np.testing.assert_allclose(fbp_2D_parallel.as_array(),self.golden_data_cs.as_array(), atol=0.6)

        #3D parallel
        ag = self.ag_parallel
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='gpu')
        fp_3D = A.direct(self.golden_data)

        fbp = FBP(ig, ag, 'gpu')
        fbp.set_input(fp_3D)
        fbp_3D_parallel = fbp.get_output()
        np.testing.assert_allclose(fbp_3D_parallel.as_array(),self.golden_data.as_array(),atol=0.6)
    
    @unittest.skipUnless(has_astra, "Astra not built with CUDA")
    def test_FBPcpu(self):
        #2D parallel
        ag = self.ag_parallel.subset(vertical='centre')
        ig = ag.get_ImageGeometry()
        A = ProjectionOperator(ig, ag, device='cpu')
        fp_2D = A.direct(self.golden_data_cs)

        fbp = FBP(ig, ag, 'cpu')
        fbp.set_input(fp_2D)
        fbp_2D_parallel = fbp.get_output()

        np.testing.assert_allclose(fbp_2D_parallel.as_array(),self.golden_data_cs.as_array(),atol=0.6)

