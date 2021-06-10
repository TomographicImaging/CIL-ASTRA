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
from cil.plugins.astra.processors import AstraForwardProjector2D, AstraBackProjector2D



class AstraProjector2D(LinearOperator):
    r'''AstraProjector22D wraps ASTRA 2D Projectors for CPU and GPU.
    
    :param image_geometry: The CIL ImageGeometry object describing your reconstruction volume
    :type image_geometry: ImageGeometry
    :param acquisition_geometry: The CIL AcquisitionGeometry object describing your sinogram data
    :type acquisition_geometry: AcquisitionGeometry
    :param device: The device to run on 'gpu' or 'cpu'
    :type device: string
    '''
    def __init__(self, image_geometry, acquisition_geometry, device):
        super(AstraProjector2D, self).__init__(image_geometry, range_geometry=acquisition_geometry)
        
        self.fp = AstraForwardProjector2D(volume_geometry=image_geometry,
                                        sinogram_geometry=acquisition_geometry,
                                        proj_id = None,
                                        device=device)
        
        self.bp = AstraBackProjector2D(volume_geometry = image_geometry,
                                        sinogram_geometry = acquisition_geometry,
                                        proj_id = None,
                                        device = device)
                           
        
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




