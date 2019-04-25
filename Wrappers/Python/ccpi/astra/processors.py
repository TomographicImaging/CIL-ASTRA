from ccpi.framework import DataProcessor, ImageData, AcquisitionData
from ccpi.astra.utils import convert_geometry_to_astra
import astra


class AstraForwardProjector(DataProcessor):
    '''AstraForwardProjector
    
    Forward project ImageData to AcquisitionData using ASTRA proj_id.
    
    Input: ImageData
    Parameter: proj_id
    Output: AcquisitionData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_id=None,
                 device='cpu'):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'device'  : device
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraForwardProjector, self).__init__(**kwargs)
        
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
        if dataset.number_of_dimensions == 3 or\
           dataset.number_of_dimensions == 2:
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
    
    def process(self):
        IM = self.get_input()
        DATA = AcquisitionData(geometry=self.sinogram_geometry)
        #sinogram_id, DATA = astra.create_sino( IM.as_array(), 
        #                           self.proj_id)
        sinogram_id, DATA.array = astra.create_sino(IM.as_array(), 
                                                           self.proj_id)
        astra.data2d.delete(sinogram_id)
        
        if self.device == 'cpu':
            return DATA
        else:
            if self.sinogram_geometry.geom_type == 'cone':
                return DATA
            else:
                 scaling = 1.0/self.volume_geometry.voxel_size_x
                 return scaling*DATA

class AstraBackProjector(DataProcessor):
    '''AstraBackProjector
    
    Back project AcquisitionData to ImageData using ASTRA proj_id.
    
    Input: AcquisitionData
    Parameter: proj_id
    Output: ImageData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_id=None,
                 device='cpu'):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_id'  : proj_id,
                  'device'  : device
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraBackProjector, self).__init__(**kwargs)
        
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
    
    def process(self):
        DATA = self.get_input()
        IM = ImageData(geometry=self.volume_geometry)
        rec_id, IM.array = astra.create_backprojection(DATA.as_array(),
                            self.proj_id)
        astra.data2d.delete(rec_id)
        
        if self.device == 'cpu':
            return IM
        else:
            scaling = self.volume_geometry.voxel_size_x**3  
            return scaling*IM

class AstraForwardProjectorMC(AstraForwardProjector):
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
    def process(self):
        IM = self.get_input()
        #create the output AcquisitionData
        DATA = AcquisitionData(geometry=self.sinogram_geometry)
        
        for k in range(DATA.geometry.channels):
            sinogram_id, DATA.as_array()[k] = astra.create_sino(IM.as_array()[k], 
                                                           self.proj_id)
            astra.data2d.delete(sinogram_id)
        
        if self.device == 'cpu':
            return DATA
        else:
            if self.sinogram_geometry.geom_type == 'cone':
                return DATA
            else:
                 scaling = (1.0/self.volume_geometry.voxel_size_x) 
                 return scaling*DATA

class AstraBackProjectorMC(AstraBackProjector):
    '''AstraBackProjector Multi channel
    
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
    def process(self):
        DATA = self.get_input()
        
        IM = ImageData(geometry=self.volume_geometry)
        
        for k in range(IM.geometry.channels):
            rec_id, IM.as_array()[k] = astra.create_backprojection(
                    DATA.as_array()[k], 
                    self.proj_id)
            astra.data2d.delete(rec_id)
        
        if self.device == 'cpu':
            return IM
        else:
            scaling = self.volume_geometry.voxel_size_x**3  
            return scaling*IM

class AstraForwardProjector3D(DataProcessor):
    '''AstraForwardProjector3D
    
    Forward project ImageData to AcquisitionData using ASTRA proj_geom and 
    vol_geom.
    
    Input: ImageData
    Parameter: proj_geom, vol_geom
    Output: AcquisitionData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None,
                 output_axes_order=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom,
                  'output_axes_order'  : output_axes_order
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraForwardProjector3D, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # Also store ASTRA geometries
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
    
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions 3, got {0}"\
                             .format(dataset.number_of_dimensions))
    
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
    
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def set_vol_geom(self, vol_geom):
        self.vol_geom = vol_geom
    
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self):
        IM = self.get_input()
        DATA = AcquisitionData(geometry=self.sinogram_geometry,
                               dimension_labels=self.output_axes_order)
        sinogram_id, DATA.array = astra.create_sino3d_gpu(IM.as_array(), 
                                                           self.proj_geom,
                                                           self.vol_geom)
        astra.data3d.delete(sinogram_id)
        # 3D CUDA FP does not need scaling
        return DATA

class AstraBackProjector3D(DataProcessor):
    '''AstraBackProjector3D
    
    Back project AcquisitionData to ImageData using ASTRA proj_geom, vol_geom.
    
    Input: AcquisitionData
    Parameter: proj_geom, vol_geom
    Output: ImageData
    '''
    
    def __init__(self,
                 volume_geometry=None,
                 sinogram_geometry=None,
                 proj_geom=None,
                 vol_geom=None,
                 output_axes_order=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : proj_geom,
                  'vol_geom'  : vol_geom,
                  'output_axes_order'  : output_axes_order
                  }
        
        #DataProcessor.__init__(self, **kwargs)
        super(AstraBackProjector3D, self).__init__(**kwargs)
        
        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)
        
                # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.volume_geometry,
                                                        self.sinogram_geometry)
        
        # Also store ASTRA geometries
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom
    
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3:
               return True
        else:
            raise ValueError("Expected input dimensions is 3, got {0}"\
                             .format(dataset.number_of_dimensions))
        
    def set_ImageGeometry(self, volume_geometry):
        self.volume_geometry = volume_geometry
        
    def set_AcquisitionGeometry(self, sinogram_geometry):
        self.sinogram_geometry = sinogram_geometry
    
    def process(self):
        DATA = self.get_input()
        IM = ImageData(geometry=self.volume_geometry,
                       dimension_labels=self.output_axes_order)
        rec_id, IM.array = astra.create_backprojection3d_gpu(DATA.as_array(),
                            self.proj_geom,
                            self.vol_geom)
        astra.data3d.delete(rec_id)
        
        # Scaling of 3D ASTRA backprojector, works both parallel and cone.
        scaling = 1/self.volume_geometry.voxel_size_x**2  
        return scaling*IM
