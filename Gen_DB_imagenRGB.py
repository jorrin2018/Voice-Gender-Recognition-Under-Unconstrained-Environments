# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:42:48 2023

Construcciòn de base de datos, generando imagenes de MelSpectrograma 
y guardando en el archivo en directorio de base de datos

dataset/baseMelSpec
                  |
                  |--- test 
                         |
                         |- Ae. aegypti   --- 500 imagenes 
                         |- Ae. albopictus  --- 500 imagenes 
                         |- An. arabienses  --- 500 imagenes
                         |- An. gambiae     --- 500 imagenes
                         |- C. pipiens     --- 500 imagenes
                         |- C. quinquefasciatus --- 500 imagenes
                 |--- train 
                       |
                       |- Ae. aegypti   --- 7000 imagenes 
                       |- Ae. albopictus  --- 7000 imagenes 
                       |- An. arabienses  --- 7000 imagenes
                       |- An. gambiae     --- 7000 imagenes
                       |- C. pipiens     --- 7000 imagenes
                       |- C. quinquefasciatus --- 7000 imagenes
                        
                                

@author: Mariko Nakano 

"""
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

###### Ruta de datos ######
### Ruta de datos de sonido
sound_path= '../dataset/Sonido/'
### Ruta de base de datos que va a generar
dataset_path = '../dataset/baseMelSpec/'

#### Características sonido
sr = 16000
n_data = 5000   # numero de datos

##### Parametros de Melspectrograma
n_fft = 400
win_length = n_fft   # predeterminado 
hop_length = 22
window = "hann" # predeterminado
n_mels =224
n_tbin = int(n_data/hop_length)+1

##### Determinar el numero de datos de entrenamiento y el número de datos de prueba
Num_Train = 7000
Num_Test = 500

#### Inspeccionar si existe directory y si no existe este directory se genera
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(os.path.join(dataset_path,"train"), exist_ok=True)
os.makedirs(os.path.join(dataset_path,"test"), exist_ok=True)

especie = ["Ae. aegypti", "Ae. albopictus", "An. arabiensis","An. gambiae","C. pipiens","C. quinquefasciatus"]

#Generar una lista de 2D vacia
file_lists =[[] for i in range(len(especie))]

#file_lists[0] : lista de todos los archivos de Ae. aegypti  15000
#file_lists[1] : lista de todos los archivos de Ae. albopictus 15000
#file_lists[0] : lista de todos los archivos de An. arabiensis 7500
#file_lists[0] : lista de todos los archivos de Ae. gambiae    7500
#file_lists[0] : lista de todos los archivos de C. pipiens     7500
#file_lists[0] : lista de todos los archivos de C. quiquefasciatus  7500

# Aleatoriamente obtener 7000 archivos de cada especie
np.random.seed(seed=0)
for i in range(len(especie)):
    list_especie = os.listdir((os.path.join(sound_path, especie[i])))
    np.random.shuffle(list_especie)
    file_lists[i].append(list_especie)
    
#### Generar Imagenes de entrenamiento 
plt.rcParams["figure.figsize"] = [3.50, 3.50]   # imagen cuadrada es más conveniente
plt.rcParams["figure.autolayout"] = True

#### Función para generar imagen y guardar en archivo correspondiente
def save_SPimage(spec, i, sr, mode, file):
    ### Generar directorio que corresponde a indice i
    ruta = os.path.join(dataset_path,mode,especie[i])
    os.makedirs(ruta, exist_ok=True)
    
    ### configuracion de figuras ####
    fig, ax = plt.subplots()
    img=librosa.display.specshow(spec,sr=sr)  # Revisar que tipo de seudocolor proporciona mejor resultados
    file = file+".jpg"
    plt.savefig(os.path.join(ruta,file), dpi=90, bbox_inches ="tight", pad_inches =0)  # Esta configuración es más conveniente
    #plt.show()
    plt.close()

#### Usando los primeros 7000 archivos de cada especie, generar datos de entrenamiento

print(" Construccion de base de datos de imagenes")
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
     

