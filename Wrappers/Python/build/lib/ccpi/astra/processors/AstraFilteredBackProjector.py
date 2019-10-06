from ccpi.framework import DataProcessor, ImageData
from ccpi.astra.utils import convert_geometry_to_astra
import astra


class AstraFilteredBackProjector(DataProcessor):
    
    '''AstraFilteredBackProjector
    
    Filtered Back project AcquisitionData to ImageData using ASTRA proj_id.
        
    This is for 2D data. FBP with CPU has only ram-lak filter
    
    Input: AcquisitionData
    Parameter: proj_id
    FilterType: none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    # blackman-nuttall, flat-top, kaiser, parzen
    Output: ImageData
    
    '''
    
    def __init__(self,
                 volume_geometry = None,
                 sinogram_geometry = None,
                 proj_id = None,
                 filter_type = None,
                 device = 'cpu'):
        
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'filter_type'  : filter_type,
                  'device'  : device
                  }
        
        super(AstraFilteredBackProjector, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            
            # Note that 'line' only one option
            
            if self.sinogram_geometry.geom_type == 'parallel':
                self.set_projector(astra.create_projector('line', proj_geom, vol_geom) )
                
            elif self.sinogram_geometry.geom_type == 'cone':
                self.set_projector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
                
            else:
                NotImplemented 
                
        elif device == 'gpu':
            
            self.set_projector(astra.create_projector('cuda', proj_geom, vol_geom) )
            
        else:
            NotImplemented
            
    
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
               return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def set_projector(self, proj_id):
        self.proj_id = proj_id
        
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def set_filter(self, filter_type):
        self.filter_type = filter_type
    
    def process(self):
        
        if self.device == 'cpu' and self.filter_type !='ram-lak':
            raise NotImplementedError('Currently in astra, 2D FBP is using only the ram-lak filter, switch to gpu for other filters')

        DATA = self.get_input()
        IM = self.volume_geometry.allocate()
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
                     
        # Create a data object for the reconstruction
        rec_id = astra.data2d.create( '-vol', vol_geom)
           
        # Create a data object to hold the sinogram data
        sinogram_id = astra.data2d.create('-sino', proj_geom, DATA.as_array())
            
        # Create configuration for reconstruction algorithm
        if self.device == 'cpu':
            cfg = astra.astra_dict('FBP')
            cfg['ProjectorId'] = self.proj_id
        else:
            cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        
        # possible values for FilterType:
        # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
        # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
        # blackman-nuttall, flat-top, kaiser, parzen
        cfg['FilterType'] = self.filter_type
        
        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
            
        # Get the result
        IM.array = astra.data2d.get(rec_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
    
#        elif self.sinogram_geometry.geom_type == 'cone':
#            
#            rec_id = astra.data3d.create('-vol', vol_geom)
#            sinogram_id = astra.create_sino3d_gpu(DATA.as_array(), proj_geom, vol_geom)
#
#            cfg = astra.astra_dict('FDK_CUDA')
#            cfg['ReconstructionDataId'] = rec_id
#            cfg['ProjectionDataId'] = sinogram_id
#            alg_id = astra.algorithm.create(cfg)
#            astra.algorithm.run(alg_id)
#            
#            IM.array = astra.data3d.get(rec_id)
#            astra.data3d.delete(rec_id)
#            astra.data3d.delete(sinogram_id)
                    
        astra.algorithm.delete(alg_id)

        if self.device == 'cpu':
            return IM
        else:
            scaling = self.volume_geometry.voxel_size_x**3
            return scaling*IM
