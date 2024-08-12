import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from tqdm import tqdm

class Hologram:
    def __init__(self, Path: str, 
                 iterations: int,
                 path_to_save: str,
                 device: torch.device = 'cpu',
                 save: bool = False,
                 preview: bool = False) -> None:
        """
        Create Holograms based on 8-bit PNG, JPEG and BMP set of images or image. 

        Parameters

        * Path : str

        * iterations : int
            Number of iterations in the Gerchberg-Saxton Algorithm

        * path_to_save : str

        * device : torch.device
            Device where it is going to be processed

        * save : bool

        * preview : Plot the hologram
        """

        self.filepath = Path
        self.projections = self.load_images()
        self.device = device
        self.iterations = iterations
        self.path_to_save = path_to_save

        self.Holograms = []

        for i in tqdm(range(self.projections.shape[0]), desc = 'Converting projections into holograms...'):
            self.Phase = self.Gerchberg_Saxton(projection = self.projections[i, :, :])
            Temp_hologram = self.Binarization().cpu()
            self.Holograms.append(Temp_hologram)
            if save:
                self.save(Temp_hologram, i)

        if preview:
            num = randint(0,self.projections.shape[0])
            plt.imshow(torch.log(torch.abs(torch.fft.fft2(self.Holograms[num])) ** 2) + 1, cmap= 'gray')
            plt.axis('off')
            plt.show()


    def load_images(self):
        image_list = []
        # Iterate through all files in the folder
        for filename in sorted(os.listdir(str(self.filepath))):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Construct the full file path
                file_path = os.path.join(self.filepath, filename)
                    
                # Open the image using PIL
                img = Image.open(file_path)
                    
                    
                # Convert the image to a NumPy array
                img_array = np.array(img)
                    
                # Append the array to the list
                image_list.append(img_array)
            
        # Convert the list of arrays to a NumPy array
        images_tensor = torch.tensor(np.array(image_list))
            
        return images_tensor

    def Gerchberg_Saxton(self, projection):
        Amplitude = projection.to(self.device)
        A = torch.fft.ifft2(Amplitude).to(self.device)
        for _ in range(self.iterations):
            B = abs(1.0) * torch.exp(1j * torch.angle(A))
            C = torch.fft.fft2(B)
            D = torch.abs(Amplitude) * torch.exp(1j * torch.angle(C))
            A = torch.fft.ifft2(D)

        return torch.angle(A)
    
    def Binarization(self):
        V = abs(1.0) * torch.exp(1j * self.Phase)
        Amp = torch.abs(V) / torch.max(torch.abs(V))
        Phi = torch.angle(V)
        pp = torch.asin(Amp)
        qq = Phi

        return 0.5 + 0.5*torch.sign(torch.cos(pp) + torch.cos(qq))

    
    def save(self, hologram, i):
        Image.fromarray(hologram.numpy() * 255).convert('1').save(os.path.join(self.path_to_save, f'{i:04}.png'))
