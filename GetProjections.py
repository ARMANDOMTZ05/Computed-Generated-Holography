import vamtoolbox as vam
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from PIL import Image

class Image_config:
    def __init__(self, sino,
                 Image_dims: tuple,
                 intensity_scale: float ,
                 size_scale: float,
                 save: bool = False,
                 angle_deg: float = 0.0):
        self.sino = sino
        self.angle_deg = angle_deg
        self.sinogram = self.sino.array
        self.N_u, self.N_v = Image_dims
        self.size_scale = size_scale
        self.intensity_scale = intensity_scale
        self.mod_sino = np.copy(self.sinogram)
        self.save = save

        if self.angle_deg != 0.0:
            self.mod_sino = self._rotate()

        if self.size_scale != 1.0:
            self.mod_sino = self._scaleSize()

        max_intensity = np.max(self.mod_sino)
        if max_intensity > 0 and max_intensity <= 1:
            self.mod_sino = self.mod_sino * 255

        if self.intensity_scale != 1.0:
            self.mod_sino = self._scaleIntensity()

        self.mod_sino = self._cropToBounds().astype(np.uint8)

        self.Image_list = []

        for i in range(self.sinogram.shape[1]):
            image_out = np.zeros((self.N_v, self.N_u), dtype=np.uint8)
            arr = self._insertImage(image = self.mod_sino[:,i,:].T, image_out= image_out)
            self.Image_list.append(arr)
            if self.save:
                Image.fromarray(arr, mode= 'L').save(f'./Example1/{i:04}.png')

    def _scaleSize(self):
        new_height = int(self.mod_sino.shape[0]*self.size_scale)
        new_width = int(self.mod_sino.shape[2]*self.size_scale)
        mod_sinogram = np.zeros((new_height,self.sinogram.shape[1],new_width))

        for i in range(self.sinogram.shape[1]):
            mod_sinogram[:,i,:] = cv2.resize(self.mod_sino[:,i,:],(new_width,new_height),interpolation=cv2.INTER_LINEAR)

        return mod_sinogram

    def _rotate(self):
        return ndimage.rotate(self.mod_sino,self.angle_deg,axes=(0,2),reshape=True,order=1)

    def _scaleIntensity(self):
        return np.minimum(self.mod_sino*self.intensity_scale, 255)

    def _insertImage(self, image, image_out):

        S_u = image.shape[1]
        S_v = image.shape[0]
    
        u1, u2 = int(self.N_u/2 - S_u/2), int(self.N_u/2 + S_u/2)
        v1, v2 = int(self.N_v/2), int(self.N_v/2 + S_v)

        if u1 < 0 or u2 > self.N_u:
            raise Exception("Image could not be inserted because it is either too large in the u-dimension or the offset causes it to extend out of the input screen size")
        if v1 < 0 or v2 > self.N_v:
            raise Exception("Image could not be inserted because it is either too large in the v-dimension or the offset causes it to extend out of the input screen size")
            
        image_out[v1:v2,u1:u2] = image


        return image_out
    
    def _cropToBounds(self):
        collapsed_sinogram = np.squeeze(np.sum(self.mod_sino,1))

        collapsed_u_sinogram = np.sum(collapsed_sinogram,1)
        collapsed_v_sinogram = np.sum(collapsed_sinogram,0)

        indices_u = np.argwhere(collapsed_u_sinogram != 0.0)
        indices_v = np.argwhere(collapsed_v_sinogram != 0.0)
        first_u, last_u = int(indices_u[0]), int(indices_u[-1])
        first_v, last_v = int(indices_v[0]), int(indices_v[-1])

        # if first_u == 0 or last_u == sinogram.shape[0]:
        #     mod_sinogram = sinogram
        # elif first_v == 0 or last_v == sinogram.shape[2]:
        #     mod_sinogram = sinogram
        # else:
        #     mod_sinogram = sinogram[first_v:last_v,:,first_u:last_u]
        mod_sinogram = self.mod_sino[first_u:last_u,:,first_v:last_v]
        
        return mod_sinogram


sino = vam.geometry.loadVolume(file_name=r'.\Optics_Test.sino')

iconfig = Image_config(sino = sino,
                       Image_dims=(1920, 1080),
                       intensity_scale= 2.0,
                       size_scale= 2.0,
                       save = False)

plt.plot(iconfig.Image_list[90][539,:])
#plt.colorbar()
plt.show()