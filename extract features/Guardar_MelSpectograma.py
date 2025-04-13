# -*- coding: utf-8 -*-
"""
Script para la generación y almacenamiento de Espectrogramas Mel
Este script toma archivos de audio WAV y genera sus representaciones
en forma de espectrogramas Mel, guardándolos como imágenes.

Características principales:
- Procesa audios del dataset VoxCeleb
- Genera espectrogramas Mel normalizados
- Guarda las imágenes en formato JPG
- Organiza las imágenes en carpetas según género (male/female)

@author: Mariko Nakano 
"""

# Importación de librerías necesarias
import librosa  # Para procesamiento de audio
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización
from PIL import Image  # Para manejo de imágenes
import os  # Para operaciones de sistema de archivos

# Definición de rutas para archivos de entrada y salida
carpeta_raiz_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/train/female'
carpeta_raiz_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/train/male'
output_folder_female = 'E:/JORRIN/TESIS/DATASET/IMAGENES/probe/melspectograma/test/female'
output_folder_male = 'E:/JORRIN/TESIS/DATASET/IMAGENES/probe/melspectograma/male'

def hacer_espectograma(carpeta_raiz, output_folder):
    """
    Función principal para generar y guardar espectrogramas Mel
    
    Parámetros:
    carpeta_raiz (str): Ruta a la carpeta con los archivos de audio
    output_folder (str): Ruta donde se guardarán las imágenes generadas
    """
    # Parámetros de la señal de audio
    sr = 16000  # Frecuencia de muestreo
    n_data = 48000  # Número de muestras
    
    # Parámetros para el cálculo del espectrograma Mel
    n_fft = 1024  # Tamaño de la ventana FFT
    win_length = n_fft  # Longitud de la ventana de análisis
    hop_length = 160  # Tamaño del salto entre ventanas
    window = "hann"  # Tipo de ventana
    n_mels = 224  # Número de bandas Mel
    center = False
    n_tbin = int(n_data/hop_length)+1
    
        
    for elemento in os.listdir(carpeta_raiz):
        elemento1=elemento[:-4]
        elemento1=int(elemento1)
        if elemento1>5000 and elemento1<5502:
            
            # nombre de archivo
            path = carpeta_raiz +'/'+ elemento
            signal, sr = librosa.load(path, sr=sr)
            
            melspec = librosa.feature.melspectrogram(y=signal,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     win_length=win_length, 
                                                     sr=sr,
                                                     n_mels=n_mels,
                                                     window = window,
                                                     power=2)
            
            log_mspec = librosa.power_to_db(abs(melspec), ref=np.max)
            
            # Calcula la energía de cada cuadro en el espectrograma
            energia = np.sum(np.abs(log_mspec) ** 2, axis=0)
            
            # Aplica la normalización por energía
            espectrograma_normalizado = log_mspec / np.sqrt(energia)
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
            print("Haciendo melespectograma"+ ' ' + elemento )
            ##### Leer imágen que guardo #######
            # I = Image.open("out1.jpg")
            # plt.imshow(I)
            # plt.show()
            
            '''
            import os
            dataset_path =  "../dataset/Sonido/"         
            
            
            #loop through all the genres
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
                print(i)
            #ensure that we're not at the root level
                if dirpath is not dataset_path:
                    #save the semantic label
                    dirpath_components = dirpath.split('/')#genre/aedes -> ['genre','aedes]
                    semantic_label = dirpath_components[-1] #Nombre de especie de mosquitos
                    print('\n Processing {}'.format(semantic_label))
                    
                    #process files for a specific genre
                    for f in filenames[:3000]:   # Obtener 3000 sonidos de mosquitos de 6 diferentes especies
                        #load audio file
                        file_path = os.path.join(dirpath, f)
                        signal, sr = librosa.load(file_path,sr=sr)
                    
                        melspec = librosa.feature.melspectrogram(y=signal,
                                                                 n_fft=n_fft,
                                                                 hop_length=hop_length,
                                                                 win_length=win_length, 
                                                                 sr=sr,
                                                                 n_mels=n_mels,
                                                                 window = window,
                                                                 power=2)
                        log_mspec = librosa.power_to_db(abs(melspec), ref=np.max)
                        
                        data =np.append(data,np.array([log_mspec]), axis=0)
                        label = np.append(label, i-1)
                        
                        
            data = data[1:]  # primera matriz de todo con cero, hay que eliminar esta matriz
            label = label[1:] 
            
            # datos son valores reales negativos y positivos
            # label son enteros desde 0 (Ae. aegypti) hasta 5 (Culex quinquefasciatus )
            
            #np.savez(file_path_npz, data=data, label=label)
            
            '''
#hacer_espectograma(carpeta_raiz_male, output_folder_male)
hacer_espectograma(carpeta_raiz_female, output_folder_female)
#hacer_espectograma(carpeta_raiz_test_male, output_folder_test_male)

#hacer_espectograma(carpeta_raiz_test_female, output_folder_test_female)