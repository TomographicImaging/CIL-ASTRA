from ccpi.framework import DataProcessor, ImageGeometry, AcquisitionGeometry
from ccpi.astra.utils import convert_geometry_to_astra
import astra
import numpy as np

class FBP(DataProcessor):
    
    '''Filtered Back Projection is a reconstructor
    
    Input: Volume Geometry
           Sinogram Geometry
           Filter_type
           Device = cpu/gpu. For 2D cases we have the option cpu/gpu 
                             For 3D cases we have the option of gpu
                             
    
    Example:  FBP(ig, ag, 'ram-lak', 'cpu')
              FBP.set_input(sinogram)
              reconstruction = FBP.get_ouput()
                         
    Output: ImageData                             

    
    '''
    
    def __init__(self, volume_geometry, 
                       sinogram_geometry, 
                       device = 'cpu', 
                       filter_type = 'ram-lak', 
                       **kwargs): 
        
        # For 3D FBP and parallel goemetry, we perform 2D FBP per slice
        # Therofore, we need vol_geom2D, porj_geom2D attributes to setup in 
        # the DataProcessor
        
        # For 3D cone we do not need proj_id
        
        if sinogram_geometry.dimension == '3D':
            
            if sinogram_geometry.geom_type=='parallel':
                
                super(FBP, self).__init__(volume_geometry = volume_geometry, 
                                     sinogram_geometry = sinogram_geometry,
                                     device = device, proj_id = None,
                                     vol_geom2D = None, proj_geom2D = None,
                                     filter_type = filter_type)
                
            elif sinogram_geometry.geom_type=='cone':
                
                super(FBP, self).__init__(volume_geometry = volume_geometry, 
                     sinogram_geometry = sinogram_geometry,
                     device = device,
                     filter_type = filter_type)          
        else:
            
            super(FBP, self).__init__(volume_geometry = volume_geometry, 
                                      sinogram_geometry = sinogram_geometry,
                                      device = device, proj_id = None,
                                      filter_type = filter_type)
            
                                                            
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        if self.sinogram_geometry.dimension == '2D':
            
            # Here we define proj_id that will be used for ASTRA
            # ASTRA projector, to be stored
            if self.device == 'cpu':
                
                # Raise an error if the user select a filter other than ram-lak, for 2D case
                if self.filter_type !='ram-lak':
                    raise NotImplementedError('Currently in astra, 2D FBP is using only the ram-lak filter, switch to gpu for other filters')
        
                # Note that 'line' only one option            
                if self.sinogram_geometry.geom_type == 'parallel':
                    self.set_projector(astra.create_projector('line', proj_geom, vol_geom) )
                    
                elif self.sinogram_geometry.geom_type == 'cone':
                    self.set_projector(astra.create_projector('line_fanflat', proj_geom, vol_geom) )
                    
                else:
                    NotImplemented 
                    
            elif self.device == 'gpu':
                            
                self.set_projector(astra.create_projector('cuda', proj_geom, vol_geom) )
                
            else:
                NotImplemented
                
        if self.sinogram_geometry.dimension == '3D':  
            
            # Force gpu option
            self.device == 'gpu'
            
            #Raise an error if the user select a filter other than ram-lak
            if self.filter_type !='ram-lak':
                raise NotImplementedError('Currently in astra, FDK has only ram-lak available')   

            if self.sinogram_geometry.geom_type == 'parallel':   
                
                # we need a 2D image and acquisition geometry
                ig2D = ImageGeometry(voxel_num_x = self.volume_geometry.voxel_num_x,
                                          voxel_num_y = self.volume_geometry.voxel_num_y,
                                          voxel_size_x = self.volume_geometry.voxel_size_x,
                                           voxel_size_y = self.volume_geometry.voxel_size_y)
            
                ag2D = AcquisitionGeometry(geom_type = self.sinogram_geometry.geom_type,
                                                dimension = '2D',
                                                angles = self.sinogram_geometry.angles,
                                                pixel_num_h=self.sinogram_geometry.pixel_num_h,
                                                pixel_size_h=self.sinogram_geometry.pixel_size_h)
                
                
                
                # Convert to Astra Geometries                    
                self.vol_geom2D, self.proj_geom2D = convert_geometry_to_astra(ig2D,ag2D)                
                
                
                self.set_projector(astra.create_projector('cuda', self.proj_geom2D, self.vol_geom2D) )                
                
                                            
    def check_input(self, dataset):
        
        if self.sinogram_geometry.dimension == '2D':
            if dataset.number_of_dimensions == 2 or dataset.geometry.channels>1:
                return True
            else:
                raise ValueError("Expected input dimensions is 2, got {0}"\
                 .format(dataset.number_of_dimensions))  
                
        elif self.sinogram_geometry.dimension=='3D':  
            if dataset.number_of_dimensions == 3 or dataset.geometry.channels>1:
                return True
            else:
                raise ValueError("Expected input dimensions is 3, got {0}"\
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
            
        DATA = self.get_input()
        IM = self.volume_geometry.allocate()
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        if DATA.geometry.channels == 1:
            
            if self.sinogram_geometry.dimension == '2D':         
                
                # Create a data object for the reconstruction and to hold sinogram for 2D
                rec_id = astra.data2d.create( '-vol', vol_geom)
                sinogram_id = astra.data2d.create('-sino', proj_geom, DATA.as_array())
                
                # ASTRA configuration for reconstruction algorithm
                if self.device == 'cpu':
                    cfg = astra.astra_dict('FBP')
                    cfg['ProjectorId'] = self.proj_id
                else:
                    cfg = astra.astra_dict('FBP_CUDA')
                cfg['ReconstructionDataId'] = rec_id
                cfg['ProjectionDataId'] = sinogram_id    
                cfg['FilterType'] = self.filter_type
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                
                # Get the result
                IM.array = astra.data2d.get(rec_id)
                astra.data2d.delete(rec_id)
                astra.data2d.delete(sinogram_id)
                astra.algorithm.delete(alg_id)
                                                        
                if self.device == 'cpu':
                    return IM / (self.volume_geometry.voxel_size_x**2)
                else:
                    scaling = self.volume_geometry.voxel_size_x
                    return scaling * IM            
                            
                
            elif self.sinogram_geometry.dimension == '3D':
                
                # Create a data object for the reconstruction and to hold sinogram for 3D
#                rec_id = astra.data3d.create('-vol', vol_geom)
#                sinogram_id = astra.data3d.create('-sino', proj_geom, DATA.as_array())
                
                # ASTRA configuration for reconstruction algorithm        
                if self.sinogram_geometry.geom_type == 'parallel':
                    
##                    # we need a 2D image and acquisition geometry
#                    ig2D = ImageGeometry(voxel_num_x = self.volume_geometry.voxel_num_x,
#                                              voxel_num_y = self.volume_geometry.voxel_num_y,
#                                              voxel_size_x = self.volume_geometry.voxel_size_x,
#                                               voxel_size_y = self.volume_geometry.voxel_size_y)
#                
#                    ag2D = AcquisitionGeometry(geom_type = self.sinogram_geometry.geom_type,
#                                                    dimension = '2D',
#                                                    angles = self.sinogram_geometry.angles,
#                                                    pixel_num_h=self.sinogram_geometry.pixel_num_h,
#                                                    pixel_size_h=self.sinogram_geometry.pixel_size_h)
##                    
#                    self.set_projector(astra.create_projector('cuda', ag2D, ig2D) )
##                    # Convert to Astra Geometries                    
#                    vol_geom2D, proj_geom2D = convert_geometry_to_astra(ig2D,ag2D)
                    
                    rec_id = astra.data2d.create( '-vol', self.vol_geom2D)
                    
                    for i in range(DATA.shape[0]):
                    
                        # For parallel beam we run 2D FBP per slice                                        
                        sinogram_id = astra.data2d.create('-sino', self.proj_geom2D, DATA.as_array()[i])
                        
                        cfg = astra.astra_dict('FBP_CUDA')                        
                        cfg['ReconstructionDataId'] = rec_id
                        cfg['ProjectionDataId'] = sinogram_id
                        cfg['FilterType'] = self.filter_type
                        alg_id = astra.algorithm.create(cfg)
                        astra.algorithm.run(alg_id)
                        
                        # since we run gpu we need to rescale
                        IM.array[i] = np.flip(astra.data2d.get(rec_id) *  self.volume_geometry.voxel_size_x,0)  
                        
                    return IM                        
                        
                    astra.algorithm.delete(alg_id)
                    astra.data2d.delete(rec_id)
                    astra.data2d.delete(sinogram_id)                         
                
                elif self.sinogram_geometry.geom_type == 'cone':
                    
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
            

        
        


 
