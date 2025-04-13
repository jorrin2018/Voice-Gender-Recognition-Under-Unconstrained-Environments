# -*- coding: utf-8 -*-
"""
Script para la generación y almacenamiento de características MFCC
Este script extrae los Coeficientes Cepstrales en las Frecuencias de Mel (MFCC)
de archivos de audio y los guarda como imágenes para su uso posterior en
tareas de clasificación de género de voz.

Características:
- Extracción de MFCC usando librosa
- Normalización y procesamiento de características
- Organización por género y conjunto de datos
- Visualización y almacenamiento optimizado

@author: MCIM
"""

import librosa  # Para procesamiento de señales de audio
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización
from PIL import Image  # Para procesamiento de imágenes
import os  # Para operaciones de sistema de archivos

# Definición de rutas para los conjuntos de datos
carpeta_raiz_train_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/female'
carpeta_raiz_train_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/male'
carpeta_raiz_test_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/female'
carpeta_raiz_test_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/male'

# Rutas para guardar los MFCC generados
output_folder_train_female = 'E:/JORRIN/TESIS/DATASET/IMAGENES/mfcc/train/female'
output_folder_train_male = 'E:/JORRIN/TESIS/DATASET/IMAGENES/mfcc/train/male'
output_folder_test_female = 'E:/JORRIN/TESIS/DATASET/IMAGENES/mfcc/test/female'
output_folder_test_male = 'E:/JORRIN/TESIS/DATASET/IMAGENES/mfcc/test/male'

def hacer_mfcc(carpeta_raiz, output_folder):
    """
    Genera y guarda los MFCC de los archivos de audio en una carpeta
    
    Args:
        carpeta_raiz (str): Ruta a la carpeta con los archivos WAV
        output_folder (str): Ruta donde se guardarán las imágenes MFCC
    """
    # Parámetros de procesamiento de audio
    sr = 16000  # Frecuencia de muestreo
    n_data = 48000  # Número de muestras
    
    # Parámetros para el cálculo de MFCC
    n_fft = 1024  # Tamaño de la ventana FFT
    win_length = n_fft  # Longitud de la ventana de análisis
    hop_length = 160  # Tamaño del salto entre ventanas
    window = "hann"  # Tipo de ventana
       
    for elemento in os.listdir(carpeta_raiz):
        if elemento.endswith('wav'):
            
            # nombre de archivo
            path = carpeta_raiz +'/'+ elemento
            signal, sr = librosa.load(path, sr=sr)
            
            #calcular mfcc
            mfcc = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr, 
            n_mfcc=224, window = window , dct_type=2)

            # Mostrar los MFCC utilizando librosa.display.specshow
            librosa.display.specshow(mfcc, x_axis='time')
            
                       
            # Calcula la energía de cada cuadro en el espectrograma
            energia = np.sum(np.abs(mfcc) ** 2, axis=0)
            
            # Aplica la normalización por energía
            espectrograma_normalizado = mfcc / np.sqrt(energia)
            ### configuracion de figuras ####
            #plt.rcParams["figure.figsize"] = [3.50, 3.50]   # imagen cuadrada es más conveniente
            plt.rcParams["figure.figsize"] = [3.5, 3.5]
            plt.rcParams["figure.autolayout"] = True
            
            fig, ax = plt.subplots()
               
            
            img=librosa.display.specshow(espectrograma_normalizado,sr=sr, cmap='inferno')
            # Generar el nombre del archivo utilizando el nombre y la carpeta de salida
            nombre_archivo = os.path.join(output_folder, f"{elemento[:-4]}.jpg")
            
            # Guardar la figura en el archivo correspondiente
            plt.savefig(nombre_archivo, dpi=90, bbox_inches ="tight", pad_inches =0)
            #plt.savefig("out1.jpg", dpi=90, bbox_inches ="tight", pad_inches =0)  # Esta configuración es más conveniente
            #plt.savefig("out1.jpg", dpi=90)  # Esta configuración es más conveniente
            #plt.show()
            plt.close()
            print("Haciendo mfcc"+ ' ' + elemento )
            ##### Leer imágen que guardo #######
            # I = Image.open("out1.jpg")
            # plt.imshow(I)
            # plt.show()
            
hacer_mfcc(carpeta_raiz_train_male, output_folder_train_male)
hacer_mfcc(carpeta_raiz_test_male, output_folder_test_male)
hacer_mfcc(carpeta_raiz_train_female, output_folder_train_female)
hacer_mfcc(carpeta_raiz_test_female, output_folder_test_female)