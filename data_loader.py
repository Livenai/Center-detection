import torch
import json
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import os
import json
import numpy as np

LIMIT = 50000
width = 400
height = 300

from torch.utils.data import Dataset

class baseDataset(Dataset):
    def __init__(self, json_file_path):
        self.labels=[]
        self.limit = LIMIT
        self.img_size = (width, height)
        self.images = [] #Aqui guardamos todas las imagenes del dataset
        self.json_file_path = json_file_path
        
        self.load_data()
    
    #Devuelve la imagen de la posicion index de la lista de imagenes a usar
    def __getitem__(self,index):
        img = self.images[index] 
        label = self.labels[index]
        return img, label  # hay que devolver también self.labels[index]
    
    def __len__(self):
        return len(self.images)

    
    def load_data(self):        
        index = 0
        json_data = json.loads(open(self.json_file_path, "rb").read())
        initial_path = '/'.join(self.json_file_path.split('/')[:-1])
        for data in json_data:
            img_name = data['imagen']





            point = data['punto']   # con estos datos se crean las etiquetas y se añaden a self.labels
            img_path = initial_path + '/out/' + img_name #'VueltaRuido_condiciones/out/' + img_name
            
            if os.path.exists(img_path):
                image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) # OJO AL astype()
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
                #image = torch.from_numpy(image).type(torch.FloatTensor)
                image = (image/255.)




                # Transformacion a tensor 
                transforms_list = [transforms.ToTensor()]
                transform = transforms.Compose(transforms_list)
                image = transform(image)
                self.images.append(image)
                ###########################



                label = torch.zeros(2)
                # Normalizamos
                label[0] = (point['X'] / width)  * 2 - 1
                label[1] = (point['Y'] / height) * 2 - 1
                self.labels.append(label)


                if index % 100 == 0:
                    print(index)
                index += 1
                
                if index > self.limit:
                    print("Stop loading data, limit reached")
                    break


"""
if __name__ == "__main__":
    dataset = baseDataset('VueltaRuido_condiciones/ConRuido_cond_clim.json')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
"""
