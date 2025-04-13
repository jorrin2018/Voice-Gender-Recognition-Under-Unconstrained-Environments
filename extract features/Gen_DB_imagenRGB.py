# -*- coding: utf-8 -*-
"""
Script para la generación de base de datos de imágenes RGB
Este script convierte archivos de audio en imágenes de espectrogramas Mel
y organiza una base de datos de imágenes estructurada para entrenamiento
y prueba de modelos de aprendizaje profundo.

Estructura de la base de datos generada:
dataset/baseMelSpec/
    ├── test/
    │   ├── male/    (500 imágenes)
    │   └── female/  (500 imágenes)
    └── train/
        ├── male/    (7000 imágenes)
        └── female/  (7000 imágenes)

@author: Mariko Nakano 
"""

# Importación de librerías necesarias
import librosa  # Para procesamiento de audio
import numpy as np  # Para operaciones numéricas
import os  # Para operaciones de sistema de archivos
import matplotlib.pyplot as plt  # Para visualización
from PIL import Image  # Para procesamiento de imágenes
from tqdm import tqdm  # Para barras de progreso

# Configuración de rutas
sound_path = '../dataset/Sonido/'  # Ruta de datos de audio
dataset_path = '../dataset/baseMelSpec/'  # Ruta de base de datos a generar

# Parámetros de procesamiento de audio
sr = 16000  # Frecuencia de muestreo
n_data = 5000  # Número de muestras por audio

# Parámetros para el cálculo del espectrograma Mel
n_fft = 400  # Tamaño de la ventana FFT
win_length = n_fft  # Longitud de la ventana de análisis
hop_length = 22  # Tamaño del salto entre ventanas
window = "hann"  # Tipo de ventana
n_mels = 224  # Número de bandas Mel
n_tbin = int(n_data/hop_length)+1  # Número de bins temporales

##### Determinar el numero de datos de entrenamiento y prueba
Num_Train = 7000  # Número de muestras para entrenamiento
Num_Test = 500  # Número de muestras para prueba

# Crear estructura de directorios
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(os.path.join(dataset_path,"train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path,"test"), exist_ok=True)

# Definir clases
especie = ["male", "female"]

# Generar lista 2D vacía para almacenar archivos
file_lists = [[] for i in range(len(especie))]

def save_SPimage(spec, i, sr, mode, file):
    """
    Guarda un espectrograma como imagen en la carpeta correspondiente
    
    Args:
        spec (ndarray): Espectrograma a guardar
        i (int): Índice de la clase
        sr (int): Frecuencia de muestreo
        mode (str): Modo ('train' o 'test')
        file (str): Nombre base del archivo
    """
    # Generar directorio correspondiente al índice i
    ruta = os.path.join(dataset_path, mode, especie[i])
    os.makedirs(ruta, exist_ok=True)
    
    # Configuración de la figura
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, sr=sr)
    file = file + ".jpg"
    plt.savefig(os.path.join(ruta,file), dpi=90, bbox_inches="tight", pad_inches=0)
    plt.close()

# Aleatorizar la selección de archivos
np.random.seed(seed=0)  # Para reproducibilidad
for i in range(len(especie)):
    list_especie = os.listdir((os.path.join(sound_path, especie[i])))
    np.random.shuffle(list_especie)
    file_lists[i].append(list_especie)

# Configuración de visualización
plt.rcParams["figure.figsize"] = [3.50, 3.50]
plt.rcParams["figure.autolayout"] = True

print(" Construcción de base de datos de imágenes")

#### Usando los primeros 7000 archivos de cada especie, generar datos de entrenamiento
'''
print(" Construccion de conjunto de entrenamiento")
for i in tqdm(range(len(especie))):   # i es indice de especie i=0: Ae.aegypti, 1: ,,,, 5
    for files in file_lists[i]:
        for file in files[:Num_Train]:  #Num_Train = 7000
            #leer archivo de sonido usando librosa
            signal, sr = librosa.load(os.path.join(sound_path,especie[i],file),sr=sr)
        
            melspec = librosa.feature.melspectrogram(y=signal,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     win_length=win_length, 
                                                     sr=sr,
                                                     n_mels=n_mels,
                                                     window = window,
                                                     power=2)
            log_mspec = librosa.power_to_db(melspec)
            # Guardar cada imagen en un archivo
            save_SPimage(log_mspec,i,sr, "train",file )
            
'''
#### Usando los ultimos 500 archivos de cada especie, generar datos de prueba
print(" Construccion de conjunto de prueba")
for i in tqdm(range(len(especie))):   # i es indice de especie i=0: Ae.aegypti, 1: ,,,, 5
    for files in file_lists[i]:
        for file in files[Num_Train:Num_Train+Num_Test]:  #Num_Tset = 7000
            #leer archivo de sonido usando librosa
            signal, sr = librosa.load(os.path.join(sound_path,especie[i],file),sr=sr)
        
            melspec = librosa.feature.melspectrogram(y=signal,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     win_length=win_length, 
                                                     sr=sr,
                                                     n_mels=n_mels,
                                                     window = window,
                                                     power=2)
            log_mspec = librosa.power_to_db(melspec)
            # Guardar cada imagen en un archivo
            save_SPimage(log_mspec,i,sr, "test",file )


