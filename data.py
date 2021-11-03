import numpy as np
from PIL import Image
#from skimage import io
import glob

import torch
from torch.utils import data


class strawberry(data.Dataset):
    
    def __init__(self, folder_path):
        
        self.image_list = glob.glob(folder_path+'*')
        self.data_len = len(self.image_list)
        

    def __getitem__(self, index):
        
        image_path = self.image_list[index]
        im = Image.open(image_path)
        im = np.transpose(np.asarray(im)/255, (2, 0, 1))
        images = torch.FloatTensor(im)

        return images

    def __len__(self):
        return self.data_len