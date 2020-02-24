from ccpi.framework import AcquisitionData_cupy as AcquisitionData

from ccpi.astra.processors import AstraForwardProjector_cupy as AstraForwardProjector

import astra


import cupy as cp

class AstraForwardProjectorMC_cupy(AstraForwardProjector):
    '''AstraForwardProjector Multi channel
    
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
        #create the output AcquisitionData
        DATA = AcquisitionData(geometry=self.sinogram_geometry)
        
#        DATA.array = cp.asnumpy(DATA.array)

        tmp_IM = cp.asnumpy(IM.as_array())
        tmp_DATA = cp.asnumpy(DATA.as_array())
        
        for k in range(DATA.geometry.channels):
            sinogram_id, tmp_DATA[k] = astra.create_sino(tmp_IM[k], 
                                                         self.proj_id)
            astra.data2d.delete(sinogram_id)
        DATA.array = cp.array(tmp_DATA)
        
        if self.device == 'cpu':
            ret = DATA
        else:
            if self.sinogram_geometry.geom_type == 'cone':
                ret = DATA
            else:
                 scaling = (1.0/self.volume_geometry.voxel_size_x) 
                 ret = scaling*DATA
        
        if out is None:
            return ret
        else:
            out.fill(ret)