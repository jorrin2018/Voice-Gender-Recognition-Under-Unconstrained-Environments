# -*- coding: utf-8 -*-
"""
Script para el procesamiento de metadatos del dataset VoxCeleb
Este script se encarga de procesar y organizar los metadatos de VoxCeleb,
incluyendo información de identidad, género y características de los videos.

Funcionalidades:
- Lectura y procesamiento de archivos JSON de metadatos
- Filtrado y organización de información por género
- Manejo de IDs de VoxCeleb y videos
- Exportación de datos procesados

@author: MCIM
"""

import numpy as np  # Para operaciones numéricas
import pandas as pd  # Para manejo de datos estructurados
import os  # Para operaciones de sistema de archivos
import json  # Para lectura de archivos JSON

def comparar_y_eliminar(df1, df2, columna1, columna2):
    """
    Compara dos DataFrames y filtra filas basándose en valores coincidentes
    entre columnas especificadas.
    
    Args:
        df1 (DataFrame): Primer DataFrame a comparar
        df2 (DataFrame): Segundo DataFrame a comparar
        columna1 (str): Nombre de la columna en df1
        columna2 (str): Nombre de la columna en df2
    
    Returns:
        DataFrame: DataFrame filtrado con las filas coincidentes
    """
    df1_filtrado = df1[df1[columna1].isin(df2[columna2])]
    return df1_filtrado

# Cargar archivos JSON con metadatos
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb_person_videos.json", "r") as cadena_json:
    voxceleb_person_videos_json = json.load(cadena_json)

with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb_video_people.json", "r") as cadena_json:
    voxceleb_video_people_json = json.load(cadena_json)  

with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb2_person_videos.json", "r") as cadena_json:
    voxceleb2_person_videos_json = json.load(cadena_json)
    
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb2_video_people.json", "r") as cadena_json:
    voxceleb2_video_people_json = json.load(cadena_json)  
    
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/yt_metadata.json", "r") as cadena_json:
    yt_metadata_json = json.load(cadena_json)  
    
# Leer archivos CSV como DataFrames
dataframe = pd.read_csv("E:/JORRIN/TESIS/DATASET/final_dataframe_extended.csv")
df_vox1 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox1_meta.csv")
df_vox2 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox2_meta.csv")
age_train = pd.read_csv("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/age-train.txt")
age_test = pd.read_csv("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/age-test.txt")

# Reemplazar guiones bajos por espacios en los nombres
age_train['Name'] = age_train['Name'].str.replace('_', ' ')
age_test['Name'] = age_test['Name'].str.replace('_', ' ')

# Filtrar DataFrame para solo quedarme con las etiquetas que me interesan    
dataframe_filtrado = dataframe[["Name", "VoxCeleb_ID", "video_id", "gender", "speaker_age"]]

# Eliminar las edades NaN
dataframe_filtrado = dataframe_filtrado.dropna()

# Eliminar duplicados
dataframe_filtrado = dataframe_filtrado.drop_duplicates(subset=["VoxCeleb_ID"])

# Restablecer los índices
dataframe_filtrado = dataframe_filtrado.reset_index(drop=True)

# Extraer columnas VoxCeleb ID y Name del DataFrame df_vox1
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

# Convertir lista de IDs a DataFrame 
df_ids = pd.DataFrame(lista_ID)
df_ids.columns = ["VoxCeleb_ID"]
df_ids1 = pd.DataFrame(lista_ID1)
df_ids1.columns = ["Name"]
df_ids2 = pd.DataFrame(lista_ID2)
df_ids2.columns = ["Genero"]

# Concatenar df_ids y df_ids1 para obtener df_vox1_final
df_vox1_final = pd.concat([df_ids, df_ids1, df_ids2], axis=1)

# Reemplazar guiones bajos por espacios en los nombres
df_vox1_final['Name'] = df_vox1_final['Name'].str.replace('_', ' ')

# Extraer columnas VoxCeleb ID y Name del DataFrame df_vox2
lista_ID = []
lista_ID1 = []
lista_ID2 = []

for i in list(range(6114)):
    texto = df_vox2.iat[i, 0]
    partes = texto.split('\t')
    if len(partes) >= 2:
        texto1 = partes[1]
        texto2 = partes[0]
        texto3 = texto.split('\t')[2]
        lista_ID.append(texto1)
        lista_ID1.append(texto2)
        lista_ID2.append(texto3)
    else:
        print(i)
    
# Convertir lista de IDs a DataFrame 
df_ids = pd.DataFrame(lista_ID)
df_ids.columns = ["VoxCeleb_ID"]
df_ids1 = pd.DataFrame(lista_ID1)
df_ids1.columns = ["Name"]
df_ids2 = pd.DataFrame(lista_ID2)
df_ids2.columns = ["Genero"]

# Concatenar df_ids y df_ids1 para obtener df_vox2_final
df_vox2_final = pd.concat([df_ids, df_ids1, df_ids2], axis=1)

# Reemplazar guiones bajos por espacios en los nombres
df_vox2_final['Name'] = df_vox2_final['Name'].str.replace('_', ' ')

# Concatenar df_vox1_final y df_vox2_final para obtener df_vox_final
df_vox_final = pd.concat([df_vox2_final, df_vox1_final], ignore_index=True)

# Ruta de los ficheros de audio
ruta_carpeta = 'E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav'

# Crear una lista para almacenar los nombres de las subcarpetas
nombres_subcarpetas = []

# Utilizar os.walk para recorrer la carpeta y sus subcarpetas
for root, dirs, files in os.walk(ruta_carpeta):
    for carpeta in dirs:
        nombres_subcarpetas.append(carpeta)

# Crear un DataFrame a partir de la lista de nombres de subcarpetas
df_audios = pd.DataFrame({'VoxCeleb_ID': nombres_subcarpetas})
df_audios = df_audios.loc[df_audios.index <= 1250]

# Exportar df_audios a CSV
nombre_archivo = 'df_audios.csv'
df_audios.to_csv(nombre_archivo, index=False)  # El argumento index=False evita que se guarde el índice en el archivo

# Buscar coincidencias entre columna Name del DataFrame test y filtrado
df_coincidencias = comparar_y_eliminar(dataframe_filtrado, age_test, "Name", "Name")

# Buscar coincidencias entre columna video_id del DataFrame test y filtrado
df_coincidencias10 = comparar_y_eliminar(dataframe_filtrado, age_test, "video_id", "video_id")

# Buscar coincidencias entre columna Name del DataFrame train y filtrado
df_coincidencias1 = comparar_y_eliminar(dataframe_filtrado, age_train, "Name", "Name")

# Buscar coincidencias entre columna VoxCeleb_ID del DataFrame df_vox_final y df_audios
df_coincidencias3 = comparar_y_eliminar(df_vox1_final, df_audios, "VoxCeleb_ID", "VoxCeleb_ID")

# Buscar coincidencias entre columna VoxCeleb_ID del DataFrame age_test y df_audios
df_coincidencias4 = comparar_y_eliminar(age_test, df_audios, "VoxCeleb_ID", "VoxCeleb_ID")

# Buscar coincidencias entre columna VoxCeleb_ID del DataFrame age_test y df_vox_final
df_coincidencias5 = comparar_y_eliminar(age_test, df_vox_final, "VoxCeleb_ID", "VoxCeleb_ID")

# Buscar coincidencias entre columna video_id del DataFrame dataframe_filtrado y df_audios
df_coincidencias6 = comparar_y_eliminar(dataframe, df_audios, "video_id", "VoxCeleb_ID")

# Buscar coincidencias entre columna Name del DataFrame age_test y df_vox_final
df_coincidencias7 = comparar_y_eliminar(age_test, df_vox_final, "Name", "Name")

# Buscar coincidencias entre columna video_id del DataFrame age_test y la columna VoxCeleb_ID df_audios
df_coincidencias8 = comparar_y_eliminar(age_test, df_audios, "video_id", "VoxCeleb_ID")

# Buscar coincidencias entre columna video_id del DataFrame age_test y la columna VoxCeleb_ID df_audios
df_coincidencias9 = comparar_y_eliminar(dataframe_filtrado, df_audios, "video_id", "VoxCeleb_ID")

# Contar cuántos elementos son femeninos
num_femeninos = len(df_vox1_final[df_vox1_final['Genero'] == 'f'])

# Contar cuántos elementos son masculinos
num_masculinos = len(df_vox1_final[df_vox1_final['Genero'] == 'm'])