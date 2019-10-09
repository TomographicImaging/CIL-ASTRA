from ccpi.framework import DataProcessor
from ccpi.astra.utils import convert_geometry_to_astra
import astra


class AstraFDK(DataProcessor):
    
    ''' FDK algorithm of Astra acting on 3D data.
        Currently only ram-lak filter is available from Astra
        
    '''
    
    def __init__(self,
                 volume_geometry = None,
                 sinogram_geometry = None,
                 filter_type = None):

        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'filter_type'  : filter_type
                  }
        
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraFDK, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        
    def check_input(self, dataset):
        if dataset.number_of_dimensions != 3:
            raise ValueError("Expected input dimensions is 3, got {0}"\
                             .format(dataset.number_of_dimensions))    
        else:
            return True
        
    def set_projector(self, proj_id):
        self.proj_id = proj_id     
        
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
        
    def set_filter(self, filter_type):
        self.filter_type = filter_type     
        
    def process(self):
        
        if self.filter_type !='ram-lak':
            raise NotImplementedError('Currently in astra, FDK has only ram-lak available')

        DATA = self.get_input()
        IM = self.volume_geometry.allocate()
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
          
        rec_id = astra.data3d.create('-vol', vol_geom)
        sinogram_id = astra.data3d.create('-sino', proj_geom, DATA.as_array())

        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
            
        IM.array = astra.data3d.get(rec_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinogram_id)
                    
        astra.algorithm.delete(alg_id)

        return IM

