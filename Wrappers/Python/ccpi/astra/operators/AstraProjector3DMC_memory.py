from ccpi.optimisation.operators import LinearOperator
from ccpi.astra.operators import AstraProjector3DSimple
import numpy as np
from ccpi.astra.utils import convert_geometry_to_astra
import astra


class AstraProjector3DMC_memory(LinearOperator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    
    def __init__(self, geomv, geomp):
        super(AstraProjector3DMC_memory, self).__init__(
            geomv, range_geometry=geomp
        )
                
        self.igtmp3D = self.domain_geometry().clone()
        self.igtmp3D.channels = 1
        self.igtmp3D.shape = self.domain_geometry().shape[1:]
        self.igtmp3D.dimension_labels = ['vertical', 'horizontal_y', 'horizontal_x']
        
        self.agtmp3D = self.range_geometry().clone()
        self.agtmp3D.channels = 1
        self.agtmp3D.shape = self.range_geometry().shape[1:]
        self.agtmp3D.dimension_labels = ['vertical', 'angle', 'horizontal']      
                   
        # Set up ASTRA Volume and projection geometry, not to be stored in self
        vol_geom, proj_geom = convert_geometry_to_astra(self.igtmp3D,
                                                        self.agtmp3D)
        
        # Also store ASTRA geometries
        self.vol_geom = vol_geom
        self.proj_geom = proj_geom   
        
                                               
        self.s1 = None
        self.channels1 = self.domain_geometry().channels
        self.channels2 = self.range_geometry().channels
        
        self.out_img = self.domain_geometry().allocate()  
        self.out_sino = self.range_geometry().allocate()        
                    
        
    def direct(self, x, out = None):
        
        if out is None:
            
            if cfg.run_with_cupy: 
                
                for i in range(self.channels1): 
                    
                    sinogram_id, tmp = astra.create_sino3d_gpu(cupy.asnumpy(x.array[i]), 
                                                           self.proj_geom,
                                                           self.vol_geom)
                    self.out_sino.array[i] = cupy.array(tmp)
                    astra.data3d.delete(sinogram_id)
                
                return self.out_sino
                
                
            else:
                
                for i in range(self.channels1):
                    
                    sinogram_id, self.out_sino.array[i] = astra.create_sino3d_gpu(x.array[i], 
                                                           self.proj_geom,
                                                           self.vol_geom)
                    astra.data3d.delete(sinogram_id)
                
                return self.out_sino
        
        else:
            
            if cfg.run_with_cupy:
                
                for i in range(self.channels1):
                    
                    sinogram_id, tmp = astra.create_sino3d_gpu(cupy.asnumpy(x.array[i]), 
                                                           self.proj_geom,
                                                           self.vol_geom)                    
#                     sinogram_id, out.array[i] = astra.create_sino3d_gpu(cupy.asnumpy(x.array[i]), 
#                                                            self.proj_geom,
#                                                            self.vol_geom)
                    out.array[i] = cupy.array(tmp)
                    astra.data3d.delete(sinogram_id)                
                
            else:
                
                for i in range(self.channels1):
                    
                    sinogram_id, out.array[i] = astra.create_sino3d_gpu(x.array[i], 
                                                           self.proj_geom,
                                                           self.vol_geom)
                    astra.data3d.delete(sinogram_id)
                         
                
        
    def adjoint(self, x, out = None):
                    
        if out is None:
            
            if cfg.run_with_cupy:
                
                for i in range(self.channels1):
                    rec_id, tmp1 = astra.create_backprojection3d_gpu(cupy.asnumpy(x.array[i]),
                                                self.proj_geom,
                                                self.vol_geom)
                    self.out_img.array[i] = cupy.array(tmp1)                
#                     rec_id, self.out_img.array[i] = astra.create_backprojection3d_gpu(cupy.asnumpy(x.array[i]),
#                                                 self.proj_geom,
#                                                 self.vol_geom)
#                     self.out_img.array[i] = cupy.array(self.out_img.array[i])
                    astra.data3d.delete(rec_id) 
                return self.out_img
                
            else:
                
                for i in range(self.channels1):
                    rec_id, self.out_img.array[i] = astra.create_backprojection3d_gpu(x.array[i],
                                                self.proj_geom,
                                                self.vol_geom)
                
                    astra.data3d.delete(rec_id)
                return self.out_img   
        
        else:
            
            if cfg.run_with_cupy:
                
                for i in range(self.channels1):
                    rec_id, tmp1 = astra.create_backprojection3d_gpu(cupy.asnumpy(x.array[i]),
                                                self.proj_geom,
                                                self.vol_geom)
                    out.array[i] = cupy.array(tmp1)
                    astra.data3d.delete(rec_id)                
                
            else:
                
                for i in range(self.channels1):
                    rec_id, out.array[i] = astra.create_backprojection3d_gpu(x.array[i],
                                                self.proj_geom,
                                                self.vol_geom)
                
                    astra.data3d.delete(rec_id)                
            

    def calculate_norm(self):
        
        self.A3D = AstraProjector3DSimple(self.igtmp3D, self.agtmp3D)          

        return self.A3D.norm()
    