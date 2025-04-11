# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 21:29:54 2023

Construir Dataset y DataLoader por su propia base de datos

@author: Mariko Nakano
"""

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

train_ruta = 'E:/JORRIN/TESIS/DATASET/IMAGENES/prueba/train'
test_ruta = 'E:/JORRIN/TESIS/DATASET/IMAGENES/prueba/test'

#### Transformations (Data Augmentation)
transform = transforms.Compose([
    transforms.Resize(256),  # Resize lo màs corto de dos lados a 256 pixeles
    transforms.CenterCrop(224),  # Obtener 224 x 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(root=train_ruta, transform = transform)
test_data = datasets.ImageFolder(root=test_ruta, transform = transform)


###### Revisión de datasets y DataLoader ####

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

### Obtener nombre de clases ####
nombre_de_Clases = train_data.classes
print('nombre de clases existentes:', nombre_de_Clases)
class_dict =train_data.class_to_idx
print('class_dict',class_dict)

# Datos (imagen  y etquetas en forma de Tensor)
img, label = train_data[8000][0], train_data[8000][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")

# Datos (imagen  y etquetas en forma de Tensor)
img, label = test_data[2500][0], test_data[2500][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")


### Construir DataLoader
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# obtener una imagen y su etiqueta, Dibujar imagen indicando su etiqueta
imagenes, labels = next(iter(train_loader))

# Revisar la forma (tamaño) de imagen y label

print(f"Image shape: {imagenes.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {labels.shape}")
print(f"label: {labels}")

# Dibujar la primera imagen de batch extraido de DataLoader
imagen = imagenes[5,:,:,:]
# para poder dibujar necesita cambiar de orden de dimensión
#   pytorech batch x canal x alto x ancho 
#    imagen canal x alto x ancho
#   pilow y pyplot  alto x ancho x canal (R, G, B)

#  Eliminar información de gradientes y convertir en numpy darray, orden alto x ancho x canal
imagen_np = imagen.detach().numpy().transpose(1,2,0) 
imagen_np = imagen_np*[0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # regresar imagen original
plt.imshow(imagen_np)
plt.axis("off")
plt.show()



