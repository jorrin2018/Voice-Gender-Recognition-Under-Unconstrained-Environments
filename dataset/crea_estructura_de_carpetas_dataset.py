# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:55:20 2023

@author: MCIM
"""
#####################################################################################
#importar librerias

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shutil

#inicializando variables
samplerate=[]
lista_elementos=[]
lista_ruta=[]
lista_ruta1=[]
lista_ruta_completa=[]


####################################################################################

#Funciones

# Función para recorrer carpetas y subcarpetas
def recorrer_carpeta(carpeta_raiz):
    
    for elemento in os.listdir(carpeta_raiz):
        ruta_completa = os.path.join(carpeta_raiz, elemento)
        if os.path.isdir(ruta_completa):
            # Si es una carpeta, recursivamente llamamos a la función en esa carpeta
            recorrer_carpeta(ruta_completa)
        else:                                                                                                                                                                                     
            # Si es un archivo, aquí puedes realizar la acción que desees
            if elemento.endswith('.wav'):
                # Procesar archivos WAV, por ejemplo, imprimir su ruta
                lista_elementos.append(elemento)
                print(f"Archivo WAV encontrado: {ruta_completa}")
     
        lista_ruta.append(ruta_completa)
        #lista_ruta1=lista_ruta[:-2]
        #lista_ruta_completa.append(lista_ruta1)
    return [lista_elementos, lista_ruta]

# Función para contar ficheros
def contar_ficheros_wav(carpeta_principal):
       
    # Inicializa un contador para la cantidad de archivos .wav
    cantidad_archivos_wav = 0
    
    # Recorre la estructura de carpetas
    for directorio_raiz, directorios, archivos in os.walk(carpeta_principal):
        for archivo in archivos:
            # Comprueba si la extensión del archivo es .wav
            if archivo.endswith('.wav'):
                cantidad_archivos_wav += 1
    
    return cantidad_archivos_wav


###############################################################################################

#leer metadata
df_vox1 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox1_meta.csv")

#obtener lista con las direcciones de los ficheros de audio
lista_ruta=os.listdir('E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav')
for i in range(1251):
    lista_ruta1.append(os.path.join('E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav',lista_ruta[i]))



###############################################################################################
#Obtener metadata con informacion de genero

#extraer columnas VoxCeleb ID, Name y genero del dataframe df_vox1
lista_ID=[]
lista_ID1=[]
lista_ID2=[]

for i in list(range(1251)):
    texto = df_vox1.iat[i,0]
    texto1 = texto.split('\t')[0]
    texto2 =texto.split('\t')[1]
    texto3 =texto.split('\t')[2]
    lista_ID.append(texto1)
    lista_ID1.append(texto2)
    lista_ID2.append(texto3)

#convertir lista de IDS a dataframe 
df_ids=pd.DataFrame(lista_ID)
df_ids.columns=["VoxCeleb_ID"]
df_ids1=pd.DataFrame(lista_ID1)
df_ids1.columns=["Name"]
df_ids2=pd.DataFrame(lista_ID2)
df_ids2.columns=["Genero"]
#concatenar df_ids y df_ids1 para obtener df_vox1_final
df_vox1_final=pd.concat([df_ids, df_ids1,df_ids2], axis=1)

#reemplazar _ por espacios
df_vox1_final['Name'] = df_vox1_final['Name'].str.replace( '_',' ')



#################################################################################
#Crear estrutura de BD

#dividir dataframe en male y female
df_male= df_vox1_final.loc[df_vox1_final['Genero'] == 'm']
df_female= df_vox1_final.loc[df_vox1_final['Genero'] == 'f']
df_male = df_male.reset_index(drop=True)
df_female = df_female.reset_index(drop=True)

#ciclo para mover los archivos male y female 
for i,ruta in enumerate(lista_ruta):
   condicion = df_male['VoxCeleb_ID'].isin([lista_ruta1[i][-7:]]) 
   condicion1 = df_female['VoxCeleb_ID'].isin([lista_ruta1[i][-7:]])
   if(condicion.any() == True):
        # mover la carpeta y sus subcarpetas a la ruta de destino
        shutil.copytree(lista_ruta1[i], 'E:/JORRIN/TESIS/DATASET/male/' + ruta)
   if(condicion1.any() == True):
         # mover la carpeta y sus subcarpetas a la ruta de destino
         shutil.copytree(lista_ruta1[i], 'E:/JORRIN/TESIS/DATASET/female/' + ruta)
   print('moviendo ficheros .wav' + '...')    

#contar utterances
utterances_male=contar_ficheros_wav('E:/JORRIN/TESIS/DATASET/BD/male')   
utterances_female=contar_ficheros_wav('E:/JORRIN/TESIS/DATASET/BD/female')   

##################################################################################################################

#---------------------------------------------------------------------------------------------------------------
#crear carpeta de entrenamiento con 56759 archivos de audio para male y 56759 archivos de audio para female
#carpeta con 


# Inicializa un contador para la cantidad de archivos .wav
cantidad_archivos_male = 0
cantidad_archivos_female = 0

#direccion de carpetas male y female

carpeta_principal_male = 'E:/JORRIN/TESIS/DATASET/BD/male/'
carpeta_principal_female = 'E:/JORRIN/TESIS/DATASET/BD/female/'
carpeta_train_wav_male='E:/JORRIN/TESIS/DATASET/AUDIOS/train/male'
carpeta_train_wav_female='E:/JORRIN/TESIS/DATASET/AUDIOS/train/female'
carpeta_train_wav_male_prueba='E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/male'
carpeta_train_wav_female_prueba='E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/female'
carpeta_test_wav_male_prueba='E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/male'
carpeta_test_wav_female_prueba='E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/female'


#crear listas con las direcciones de todos los .wav male y female 
lista_ruta=[] 
lista_elementos=[]
archivos_male=recorrer_carpeta(carpeta_principal_male)
lista_ruta=[]
lista_elementos=[]
archivos_female=recorrer_carpeta(carpeta_principal_female)

#ciclo para solo dejar las direcciones de los ficheros wav male
for i,ruta in enumerate(archivos_male[1]):
    archivos_male[1][i]=ruta.replace("\\","/")
    if archivos_male[1][i][-4:]!='.wav':
        archivos_male[1].pop(i)
    if len(archivos_male[1][i])<=60:
        archivos_male[1].pop(i)
    

#ciclo para solo dejar las direcciones de los ficheros wav female
for i,ruta in enumerate(archivos_female[1]):
    archivos_female[1][i]=ruta.replace("\\","/")
    if archivos_female[1][i][-4:]!='.wav':
        archivos_female[1].pop(i)
    if len(archivos_female[1][i])<=60:
        archivos_female[1].pop(i)
    
    
# Copiar audios male a carpeta de entrenamiento
for i,ruta in enumerate(archivos_male[1]):
    if i<=56768:#90% del total de female
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_train_wav_male
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_male, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)
     
  
    
# Copiar audios female a carpeta de entrenamiento       
for i,ruta in enumerate(archivos_female[1]):
    if i<=56768:#90% del total de female
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_train_wav_female
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_female, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)

    

# Copiar audios male a carpeta de entrenamiento de prueba con 5000 audios
for i,ruta in enumerate(archivos_male[1]):
    if i<=4999:
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_train_wav_male_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_male_prueba, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)
     
  
    
# Copiar audios female a carpeta de entrenamienti de prueba con 5000 audios   
for i,ruta in enumerate(archivos_female[1]):
    if i<=4999:
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_train_wav_female_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_female_prueba, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)
        

# Copiar audios male a carpeta de test de prueba con 5000 audios
for i,ruta in enumerate(archivos_male[1]):
    if i>=5000 and i<5500:
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_test_wav_male_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_test_wav_male_prueba, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)
     
  
    
# Copiar audios female a carpeta de test de prueba con 5000 audios   
for i,ruta in enumerate(archivos_female[1]):
    if i>=5000 and i<5500:
    # Verificar si el archivo de destino ya existe
        contador = 1
        destino=carpeta_test_wav_female_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_test_wav_female_prueba, nuevo_nombre)
            contador += 1

        shutil.copy(ruta, destino)