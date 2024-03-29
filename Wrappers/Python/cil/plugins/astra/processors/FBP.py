# -*- coding: utf-8 -*-
#  Copyright 2019 - 2022 United Kingdom Research and Innovation
#  Copyright 2019 - 2022 The University of Manchester
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

from cil.framework import DataProcessor
from cil.framework import DataOrder
from cil.plugins.astra.processors.FBP_Flexible import FBP_Flexible
from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible
from cil.plugins.astra.processors.FBP_Simple import FBP_Simple

class FBP(DataProcessor):

    '''FBP Filtered Back Projection is a reconstructor for 2D and 3D parallel and cone-beam geometries.
    It is able to back-project circular trajectories with 2 PI anglar range and equally spaced anglular steps.

    This uses the ram-lak filter
    A CPU version is provided for simple 2D parallel-beam geometries only (offsets and rotations will be ignored)
    
    Input: Volume Geometry
           Sinogram Geometry
                             
    Example:  fbp = FBP(ig, ag, device)
              fbp.set_input(data)
              reconstruction = fbp.get_ouput()
                           
    Output: ImageData                             

    
    '''
    
    def __init__(self, volume_geometry, sinogram_geometry, device='gpu'): 
        
        DataOrder.check_order_for_engine('astra', volume_geometry)
        DataOrder.check_order_for_engine('astra', sinogram_geometry) 

        if device == 'gpu':
            if sinogram_geometry.geom_type == 'parallel':
                processor = FBP_Flexible(volume_geometry, sinogram_geometry)
            else:
                processor = FDK_Flexible(volume_geometry, sinogram_geometry)
            
        else:
            UserWarning("ASTRA back-projector running on CPU will not make use of enhanced geometry parameters")

            if sinogram_geometry.geom_type == 'cone':
                raise NotImplementedError("Cannot process cone-beam data without a GPU")

            if sinogram_geometry.dimension == '2D':
                processor = FBP_Simple(volume_geometry, sinogram_geometry,  device='cpu') 
            else:
                raise NotImplementedError("Cannot process 3D data without a GPU")

        if sinogram_geometry.channels > 1: 
            raise NotImplementedError("Cannot process multi-channel data")
            #processor_full = ChannelwiseProcessor(processor, self.sinogram_geometry.channels, dimension='prepend')
            #self.processor = operator_full
        
        super(FBP, self).__init__( volume_geometry=volume_geometry, sinogram_geometry=sinogram_geometry, device=device, processor=processor)  

    def set_input(self, dataset):       
        return self.processor.set_input(dataset)

    def get_input(self):       
        return self.processor.get_input()

    def get_output(self, out=None):       
        return self.processor.get_output(out=out)

    def check_input(self, dataset):       
        return self.processor.check_input(dataset)
        
    def process(self, out=None):
        return self.processor.process(out=out)
