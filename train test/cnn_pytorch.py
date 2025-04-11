# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 22:55:09 2023

@author: MCIM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, model

# Definir transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ajusta el tamaño de los espectrogramas
    transforms.ToTensor(),  # Convierte los espectrogramas en tensores
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza los valores de píxeles
])

# Ruta de la carpeta principal que contiene las subcarpetas de clases
carpeta_principal = '/ruta/de/tu/carpeta/principal'

# Crear un conjunto de datos utilizando ImageFolder
dataset = ImageFolder(root=carpeta_principal, transform=transform)

# Crear un DataLoader para cargar los datos en lotes durante el entrenamiento
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Definir la arquitectura de la CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte los espectrogramas en tensores
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza los valores de píxeles
])

# Crea un Dataset personalizado para tus datos de espectrogramas
class SpectrogramDataset(Dataset):
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




