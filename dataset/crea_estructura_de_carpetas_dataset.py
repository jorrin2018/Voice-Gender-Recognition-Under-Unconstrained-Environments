# -*- coding: utf-8 -*-
"""
Script para la creación y organización de la estructura de carpetas del dataset
Este script gestiona la organización de los archivos de audio del dataset VoxCeleb,
separándolos en carpetas según género (male/female) y conjuntos de entrenamiento/prueba.

Funcionalidades principales:
- Lee metadata de VoxCeleb
- Organiza archivos WAV en carpetas por género
- Crea conjuntos de entrenamiento y prueba balanceados
- Mantiene un registro de la distribución de archivos

@author: MCIM
"""

# Importación de librerías necesarias
import os  # Para operaciones de sistema de archivos
import pandas as pd  # Para manejo de datos tabulares
import numpy as np  # Para operaciones numéricas
import librosa  # Para procesamiento de audio
import librosa.display  # Para visualización de audio
import matplotlib.pyplot as plt  # Para graficación
import shutil  # Para operaciones de copiado de archivos

# Inicialización de variables globales para almacenamiento de rutas y elementos
samplerate = []  # Lista de frecuencias de muestreo
lista_elementos = []  # Lista de nombres de archivos
lista_ruta = []  # Lista de rutas completas
lista_ruta1 = []  # Lista de rutas procesadas
lista_ruta_completa = []  # Lista de rutas completas procesadas

def recorrer_carpeta(carpeta_raiz):
    """
    Recorre recursivamente una carpeta y sus subcarpetas buscando archivos WAV
    
    Args:
        carpeta_raiz (str): Ruta de la carpeta a explorar
    
    Returns:
        list: Lista con [nombres de archivos WAV, rutas completas]
    """
    for elemento in os.listdir(carpeta_raiz):
        ruta_completa = os.path.join(carpeta_raiz, elemento)
        if os.path.isdir(ruta_completa):
            recorrer_carpeta(ruta_completa)
        else:
            if elemento.endswith('.wav'):
                lista_elementos.append(elemento)
                print(f"Archivo WAV encontrado: {ruta_completa}")
     
        lista_ruta.append(ruta_completa)
    return [lista_elementos, lista_ruta]

def contar_ficheros_wav(carpeta_principal):
    """
    Cuenta el número total de archivos WAV en una carpeta y sus subcarpetas
    
    Args:
        carpeta_principal (str): Ruta de la carpeta a analizar
    
    Returns:
        int: Número total de archivos WAV encontrados
    """
    cantidad_archivos_wav = 0
    for _, _, archivos in os.walk(carpeta_principal):
        for archivo in archivos:
            if archivo.endswith('.wav'):
                cantidad_archivos_wav += 1
    return cantidad_archivos_wav

# Lectura de metadatos de VoxCeleb
df_vox1 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox1_meta.csv")

# Obtener lista con las direcciones de los ficheros de audio
lista_ruta = os.listdir('E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav')
for i in range(1251):
    lista_ruta1.append(os.path.join('E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav', lista_ruta[i]))

# Obtener metadata con información de género
lista_ID = []
lista_ID1 = []
lista_ID2 = []

for i in list(range(1251)):
    texto = df_vox1.iat[i, 0]
    texto1 = texto.split('\t')[0]
    texto2 = texto.split('\t')[1]
    texto3 = texto.split('\t')[2]
    lista_ID.append(texto1)
    lista_ID1.append(texto2)
    lista_ID2.append(texto3)

# Convertir lista de IDs a dataframe
df_ids = pd.DataFrame(lista_ID)
df_ids.columns = ["VoxCeleb_ID"]
df_ids1 = pd.DataFrame(lista_ID1)
df_ids1.columns = ["Name"]
df_ids2 = pd.DataFrame(lista_ID2)
df_ids2.columns = ["Genero"]
# Concatenar df_ids y df_ids1 para obtener df_vox1_final
df_vox1_final = pd.concat([df_ids, df_ids1, df_ids2], axis=1)

# Reemplazar _ por espacios
df_vox1_final['Name'] = df_vox1_final['Name'].str.replace('_', ' ')

# Crear estructura de BD
# Dividir dataframe en male y female
df_male = df_vox1_final.loc[df_vox1_final['Genero'] == 'm']
df_female = df_vox1_final.loc[df_vox1_final['Genero'] == 'f']
df_male = df_male.reset_index(drop=True)
df_female = df_female.reset_index(drop=True)

# Ciclo para mover los archivos male y female
for i, ruta in enumerate(lista_ruta):
    condicion = df_male['VoxCeleb_ID'].isin([lista_ruta1[i][-7:]])
    condicion1 = df_female['VoxCeleb_ID'].isin([lista_ruta1[i][-7:]])
    if condicion.any() == True:
        shutil.copytree(lista_ruta1[i], 'E:/JORRIN/TESIS/DATASET/male/' + ruta)
    if condicion1.any() == True:
        shutil.copytree(lista_ruta1[i], 'E:/JORRIN/TESIS/DATASET/female/' + ruta)
    print('moviendo ficheros .wav' + '...')

# Contar utterances
utterances_male = contar_ficheros_wav('E:/JORRIN/TESIS/DATASET/BD/male')
utterances_female = contar_ficheros_wav('E:/JORRIN/TESIS/DATASET/BD/female')

# Crear carpeta de entrenamiento con 56759 archivos de audio para male y 56759 archivos de audio para female
cantidad_archivos_male = 0
cantidad_archivos_female = 0

# Dirección de carpetas male y female
carpeta_principal_male = 'E:/JORRIN/TESIS/DATASET/BD/male/'
carpeta_principal_female = 'E:/JORRIN/TESIS/DATASET/BD/female/'
carpeta_train_wav_male = 'E:/JORRIN/TESIS/DATASET/AUDIOS/train/male'
carpeta_train_wav_female = 'E:/JORRIN/TESIS/DATASET/AUDIOS/train/female'
carpeta_train_wav_male_prueba = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/male'
carpeta_train_wav_female_prueba = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/train/female'
carpeta_test_wav_male_prueba = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/male'
carpeta_test_wav_female_prueba = 'E:/JORRIN/TESIS/DATASET/AUDIOS/prueba/test/female'

# Crear listas con las direcciones de todos los .wav male y female
lista_ruta = []
lista_elementos = []
archivos_male = recorrer_carpeta(carpeta_principal_male)
lista_ruta = []
lista_elementos = []
archivos_female = recorrer_carpeta(carpeta_principal_female)

# Ciclo para solo dejar las direcciones de los ficheros wav male
for i, ruta in enumerate(archivos_male[1]):
    archivos_male[1][i] = ruta.replace("\\", "/")
    if archivos_male[1][i][-4:] != '.wav':
        archivos_male[1].pop(i)
    if len(archivos_male[1][i]) <= 60:
        archivos_male[1].pop(i)

# Ciclo para solo dejar las direcciones de los ficheros wav female
for i, ruta in enumerate(archivos_female[1]):
    archivos_female[1][i] = ruta.replace("\\", "/")
    if archivos_female[1][i][-4:] != '.wav':
        archivos_female[1].pop(i)
    if len(archivos_female[1][i]) <= 60:
        archivos_female[1].pop(i)

# Copiar audios male a carpeta de entrenamiento
for i, ruta in enumerate(archivos_male[1]):
    if i <= 56768:  # 90% del total de female
        contador = 1
        destino = carpeta_train_wav_male
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_male, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)

# Copiar audios female a carpeta de entrenamiento
for i, ruta in enumerate(archivos_female[1]):
    if i <= 56768:  # 90% del total de female
        contador = 1
        destino = carpeta_train_wav_female
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_female, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)

# Copiar audios male a carpeta de entrenamiento de prueba con 5000 audios
for i, ruta in enumerate(archivos_male[1]):
    if i <= 4999:
        contador = 1
        destino = carpeta_train_wav_male_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_male_prueba, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)

# Copiar audios female a carpeta de entrenamiento de prueba con 5000 audios
for i, ruta in enumerate(archivos_female[1]):
    if i <= 4999:
        contador = 1
        destino = carpeta_train_wav_female_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_train_wav_female_prueba, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)

# Copiar audios male a carpeta de test de prueba con 5000 audios
for i, ruta in enumerate(archivos_male[1]):
    if i >= 5000 and i < 5500:
        contador = 1
        destino = carpeta_test_wav_male_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_test_wav_male_prueba, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)

# Copiar audios female a carpeta de test de prueba con 5000 audios
for i, ruta in enumerate(archivos_female[1]):
    if i >= 5000 and i < 5500:
        contador = 1
        destino = carpeta_test_wav_female_prueba
        while os.path.exists(destino):
            nombre_base, extension = os.path.splitext(ruta[-9:])
            nuevo_nombre = f"{nombre_base}_{contador}{extension}"
            destino = os.path.join(carpeta_test_wav_female_prueba, nuevo_nombre)
            contador += 1
        shutil.copy(ruta, destino)