#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from cil.optimisation.operators import LinearOperator
from cil.plugins.astra.processors import AstraForwardProjector3D, AstraBackProjector3D

class AstraProjector3D(LinearOperator):
    r'''AstraProjector3D wraps ASTRA 3D Projectors for GPU.
    
    :param image_geometry: The CIL ImageGeometry object describing your reconstruction volume
    :type image_geometry: ImageGeometry
    :param acquisition_geometry: The CIL AcquisitionGeometry object describing your sinogram data
    :type acquisition_geometry: AcquisitionGeometry
    '''
    def __init__(self, image_geometry, acquisition_geometry):
        
        super(AstraProjector3D, self).__init__(domain_geometry=image_geometry, range_geometry=acquisition_geometry)
                    
        self.sinogram_geometry = acquisition_geometry 
        self.volume_geometry = image_geometry         
        
        self.fp = AstraForwardProjector3D(volume_geometry=image_geometry, sinogram_geometry=acquisition_geometry)       
        self.bp = AstraBackProjector3D(volume_geometry=image_geometry, sinogram_geometry=acquisition_geometry)
                      
    def direct(self, IM, out=None):
        self.fp.set_input(IM)
        
        if out is None:
            return self.fp.get_output()
        else:
            out.fill(self.fp.get_output())
    
    def adjoint(self, DATA, out=None):
        self.bp.set_input(DATA)
        
        if out is None:
            return self.bp.get_output()
        else:
            out.fill(self.bp.get_output())

    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry 

