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


from cil.plugins.astra.processors import AstraForwardProjector2D

import astra

class AstraForwardProjectorMC(AstraForwardProjector2D):
    '''AstraForwardProjector2D Multi channel
    
    Forward project ImageData to AcquisitionDataSet using ASTRA proj_id.
    
    Input: ImageDataSet
    Parameter: proj_id
    Output: AcquisitionData
    '''
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 2 or \
           dataset.number_of_dimensions == 3 or \
           dataset.number_of_dimensions == 4:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def process(self, out=None):
        IM = self.get_input()
     
        if out is None:
            DATA = self.sinogram_geometry.allocate(None)
        else:
            DATA = out

        for k in range(DATA.geometry.channels):
            vol_temp = IM.as_array()[k]
            sinogram_id, DATA.as_array()[k] = astra.create_sino(vol_temp, self.proj_id)
            astra.data2d.delete(sinogram_id)
                
        if out is None:
            return DATA
