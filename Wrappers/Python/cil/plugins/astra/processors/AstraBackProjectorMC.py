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


from cil.plugins.astra.processors import AstraBackProjector2D
import astra

class AstraBackProjectorMC(AstraBackProjector2D):
    '''AstraBackProjector2D Multi channel
    
    Back project AcquisitionData to ImageData using ASTRA proj_id.
    
    Input: AcquisitionData
    Parameter: proj_id
    Output: ImageData
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
        DATA = self.get_input()
        
        if out is None:
            IM = self.volume_geometry.allocate(None)
        else:
            IM = out

        for k in range(IM.geometry.channels):
            data_temp = DATA.as_array()[k]
            rec_id, IM.as_array()[k] = astra.create_backprojection(data_temp, self.proj_id)     
            astra.data2d.delete(rec_id)
             
        if out is None:
            return IM   
