# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 12:42:48 2023

Genera Mel_Spectrograma (db)
Displegar en la pantalla
Guardar en archivo como imagen

@author: Mariko Nakano 

"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


carpeta_raiz_train_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/female'
carpeta_raiz_train_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/male'
carpeta_raiz_test_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/female'
carpeta_raiz_test_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/male'
output_folder_train_female = 'E:/JORRIN/TESIS/DATASET/IMAGENES/melspectograma/train/female'
output_folder_train_male = 'E:/JORRIN/TESIS/DATASET/IMAGENES/melspectograma/train/male'
output_folder_test_female = 'E:/JORRIN/TESIS/DATASET/IMAGENES/melspectograma/test/female'
output_folder_test_male = 'E:/JORRIN/TESIS/DATASET/IMAGENES/melspectograma/test/male'

def hacer_espectograma(carpeta_raiz, output_folder):
    #### Parametros de datos
    sr = 16000
    n_data = 48000   # numero de datos
    ##### Parametros de Melspectrograma
    n_fft = 1024
    win_length = n_fft   # predeterminado 
    hop_length = 160
    window = "hann" # predeterminado
    n_mels =40
    center = False
    n_tbin = int(n_data/hop_length)+1
    
        
    for elemento in os.listdir(carpeta_raiz):
        if elemento.endswith('wav'):
            
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
            print("Haciendo espectograma"+ ' ' + elemento )
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
hacer_espectograma(carpeta_raiz_train_male, output_folder_train_male)
hacer_espectograma(carpeta_raiz_test_male, output_folder_test_male)
hacer_espectograma(carpeta_raiz_train_female, output_folder_train_female)
hacer_espectograma(carpeta_raiz_test_female, output_folder_test_female)