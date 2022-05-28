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
from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D


class FBP_Flexible(FDK_Flexible):

    '''FBP_Flexible Filtered Back Projection is a reconstructor for 2D and 3D parallel-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    This is a GPU version only
    
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fbp = FBP_Flexible(ig, ag)
              fbp.set_input(data)
              reconstruction = fbp.get_ouput()
                           
    Output: ImageData                                 
    '''

    def __init__(self, volume_geometry, 
                       sinogram_geometry): 
        
        super(FBP_Flexible, self).__init__( volume_geometry = volume_geometry, sinogram_geometry = sinogram_geometry)

        #convert parallel geomerty to cone with large source to object
        sino_geom_cone = sinogram_geometry.copy()
        
        #this catches behaviour modified after CIL 21.3.1 
        try:
            sino_geom_cone.config.system.align_reference_frame('cil')
        except:
            sino_geom_cone.config.system.update_reference_frame()

        #reverse ray direction unit-vector direction and extend to inf
        cone_source = -sino_geom_cone.config.system.ray.direction * sino_geom_cone.config.panel.pixel_size[1] * sino_geom_cone.config.panel.num_pixels[1] * 1e6
        detector_position = sino_geom_cone.config.system.detector.position
        detector_direction_x = sino_geom_cone.config.system.detector.direction_x

        if sino_geom_cone.dimension == '2D':
            tmp = AcquisitionGeometry.create_Cone2D(cone_source, detector_position, detector_direction_x)
        else:
            detector_direction_y = sino_geom_cone.config.system.detector.direction_y
            tmp = AcquisitionGeometry.create_Cone3D(cone_source, detector_position, detector_direction_x, detector_direction_y)

        sino_geom_cone.config.system = tmp.config.system.copy()

        self.vol_geom_astra, self.proj_geom_astra = convert_geometry_to_astra_vec_3D(volume_geometry, sino_geom_cone)
                           
    def check_input(self, dataset):
        
        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))  

        if self.sinogram_geometry.geom_type != 'parallel':
            raise ValueError("Expected input data to be parallel beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))  

        return True
        
