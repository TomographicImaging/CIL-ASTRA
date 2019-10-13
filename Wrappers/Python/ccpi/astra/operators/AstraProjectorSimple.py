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

from ccpi.optimisation.operators import LinearOperator
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector, AstraFilteredBackProjector



class AstraProjectorSimple(LinearOperator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjector(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id = None,
                                        device=device)
        
        self.bp = AstraBackProjector(volume_geometry = geomv,
                                        sinogram_geometry = geomp,
                                        proj_id = None,
                                        device = device)
                           
        # Initialise empty for singular value.
        self.s1 = None
        
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
           
    def norm(self):
        x0 = self.volume_geometry.allocate('random')
        self.s1, sall, svec = LinearOperator.PowerMethod(self, 50, x0)
        return self.s1


if __name__  == '__main__':
    
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    import numpy as np
    
    N = 30
    angles = np.linspace(0, np.pi, 180)
    ig = ImageGeometry(N, N)
    ag = AcquisitionGeometry('parallel','2D', angles, pixel_num_h = N)
    A = AstraProjectorSimple(ig, ag, 'cpu')
    print(A.norm())
    

