# -*- coding: utf-8 -*-
"""
Script de implementación de una Red Neuronal Convolucional (CNN) en PyTorch
Este script implementa un clasificador de género de voz utilizando una CNN
que procesa espectrogramas de señales de audio.

Características:
- Modelo CNN personalizado para clasificación binaria
- Carga y preprocesamiento de datos de espectrogramas
- Pipeline completo de entrenamiento y evaluación
- Soporte para GPU si está disponible

@author: MCIM
"""

import torch  # Framework principal de deep learning
import torch.nn as nn  # Módulos de redes neuronales
import torch.optim as optim  # Optimizadores
import torchvision.transforms as transforms  # Transformaciones de datos
from torchvision.datasets import ImageFolder  # Para cargar imágenes organizadas en carpetas
from torch.utils.data import DataLoader, Dataset  # Para manejo de datos

# Definir transformaciones para los datos
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionar todas las imágenes a 128x128
    transforms.ToTensor(),  # Convertir imágenes a tensores
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar valores de píxeles
])

# Ruta de la carpeta principal que contiene las subcarpetas de clases
carpeta_principal = '/ruta/de/tu/carpeta/principal'

# Crear un conjunto de datos utilizando ImageFolder
dataset = ImageFolder(root=carpeta_principal, transform=transform)

# Crear un DataLoader para cargar los datos en lotes durante el entrenamiento
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

class CNN(nn.Module):
    """
    Modelo de Red Neuronal Convolucional para clasificación de género
    
    Arquitectura:
    - Capa convolucional (1->16 canales)
    - ReLU
    - MaxPooling
    - Fully connected (16*32*32 -> 128)
    - Fully connected (128 -> num_classes)
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass del modelo
        
        Args:
            x: Tensor de entrada (batch de espectrogramas)
        
        Returns:
            Tensor con las predicciones de clase
        """
        x = self.conv1(x)  # Capa convolucional
        x = self.relu(x)   # Activación ReLU
        x = self.maxpool(x)  # Max pooling
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = self.fc1(x)  # Primera capa fully connected
        x = self.fc2(x)  # Capa de salida
        return x

class SpectrogramDataset(Dataset):
    """
    Dataset personalizado para manejar espectrogramas
    
    Args:
        data: Array de espectrogramas
        labels: Array de etiquetas
        transform: Transformaciones a aplicar
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'spectrogram': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            sample['spectrogram'] = self.transform(sample['spectrogram'])
        return sample

#Definir una función de pérdida y un optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Entrenar la CNN
num_epochs = 10

for epoch in range(num_epochs):
    for batch in data_loader:
        inputs, labels = batch['spectrogram'], batch['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}')

#Evaluar el modelo
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in data_loader:
        inputs, labels = batch['spectrogram'], batch['label']
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')




