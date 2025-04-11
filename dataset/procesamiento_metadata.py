# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:45:19 2023

@author: MCIM
"""
import numpy as np
import pandas as pd
import os
import json


#funcion para ver si un dato de un dataframe esta en otro dataframe
def comparar_y_eliminar(df1, df2, columna1, columna2):
    df1_filtrado = df1[df1[columna1].isin(df2[columna2])]
    return df1_filtrado

#leer JSON
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb_person_videos.json", "r") as cadena_json:

    voxceleb_person_videos_json=json.load(cadena_json)
    
#leer JSON
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb_video_people.json", "r") as cadena_json:

    voxceleb_video_people_json=json.load(cadena_json)  

#leer JSON
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb2_person_videos.json", "r") as cadena_json:

    voxceleb2_person_videos_json=json.load(cadena_json)
    
#leer JSON
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/voxceleb2_video_people.json", "r") as cadena_json:

    voxceleb2_video_people_json=json.load(cadena_json)  
    
#leer JSON
with open("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/yt_metadata.json", "r") as cadena_json:

    yt_metadata_json=json.load(cadena_json)  
    
    
#leer csv como dataframes
dataframe = pd.read_csv("E:/JORRIN/TESIS/DATASET/final_dataframe_extended.csv")
df_vox1 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox1_meta.csv")
df_vox2 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox2_meta.csv")
age_train = pd.read_csv("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/age-train.txt")
age_test = pd.read_csv("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/age-test.txt")


#reemplazar _ por espacios
age_train['Name'] = age_train['Name'].str.replace( '_',' ')
age_test['Name'] = age_test['Name'].str.replace( '_',' ')

#filtrar dataframe para solo quedarme con las etiquetas que me interesan    
dataframe_filtrado = dataframe[["Name","VoxCeleb_ID","video_id", "gender", "speaker_age"]]

#eliminar las edades Na
dataframe_filtrado = dataframe_filtrado.dropna()

#eliminar duplicados
dataframe_filtrado = dataframe_filtrado.drop_duplicates(subset=["VoxCeleb_ID"])

# Restablecer los índices
dataframe_filtrado = dataframe_filtrado.reset_index(drop=True)

#extraer columnas VoxCeleb ID y Name del dataframe df_vox1
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

#extraer columnas VoxCeleb ID y Name del dataframe df_vox2
lista_ID=[]
lista_ID1=[]
lista_ID2

for i in list(range(6114)):
    texto = df_vox2.iat[i,0]
    partes = texto.split('\t')
    if len(partes) >= 2:
        texto1 = partes[1]
        texto2 = partes[0]
        texto3 =texto.split('\t')[2]
        lista_ID.append(texto1)
        lista_ID1.append(texto2)
        lista_ID2.append(texto3)
    else:
        print(i)
    
#convertir lista de IDS a dataframe 
df_ids=pd.DataFrame(lista_ID)
df_ids.columns=["VoxCeleb_ID"]
df_ids1=pd.DataFrame(lista_ID1)
df_ids1.columns=["Name"]
df_ids2=pd.DataFrame(lista_ID2)
df_ids2.columns=["Genero"]

#concatenar df_ids y df_ids1 para obtener df_vox2_final
df_vox2_final=pd.concat([df_ids, df_ids1,df_ids2], axis=1)

#reemplazar _ por espacios
df_vox2_final['Name'] = df_vox2_final['Name'].str.replace( '_',' ')

#concatenar df_vox1_final y df_vox2_final para obtener df_vox_final
df_vox_final=pd.concat([df_vox2_final, df_vox1_final], ignore_index=True)

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
df_audios=df_audios.loc[df_audios.index<=1250]

#exportar df_audios a csv
# Especificar el nombre del archivo CSV en el que deseas guardar el DataFrame
nombre_archivo = 'df_audios.csv'

# Exportar el DataFrame a un archivo CSV
df_audios.to_csv(nombre_archivo, index=False)  # El argumento index=False evita que se guarde el índice en el archivo

# # Valor umbral para el índice (por ejemplo, eliminar filas con índice mayor a 2)
# valor_umbral = 1210

# # Eliminar filas con índices mayores al valor umbral
# df_audios = df_audios[df_audios.index <= valor_umbral]


#buscar coincidencias entre columna name del dataframe test y filtrado
df_coincidencias = comparar_y_eliminar(dataframe_filtrado, age_test, "Name", "Name")

#buscar coincidencias entre columna name del dataframe test y filtrado
df_coincidencias10 = comparar_y_eliminar(dataframe_filtrado, age_test, "video_id", "video_id")

#buscar coincidencias entre columna name del dataframe train y filtrado
df_coincidencias1 = comparar_y_eliminar(dataframe_filtrado, age_train, "Name", "Name")

#buscar coincidencias entre columna VoxCeleb del dataframe df_vox_final y df_audios
df_coincidencias3 = comparar_y_eliminar(df_vox1_final, df_audios, "VoxCeleb_ID", "VoxCeleb_ID")

#buscar coincidencias entre columna VoxCeleb del dataframe age_test y df_audios
df_coincidencias4 = comparar_y_eliminar(age_test, df_audios, "VoxCeleb_ID", "VoxCeleb_ID")

#buscar coincidencias entre columna VoxCeleb del dataframe age_test y df_vox_final
df_coincidencias5 = comparar_y_eliminar(age_test, df_vox_final, "VoxCeleb_ID", "VoxCeleb_ID")

#buscar coincidencias entre columna VoxCeleb del dataframe dataframe_filtrado y df_audios
df_coincidencias6 = comparar_y_eliminar(dataframe, df_audios, "video_id", "VoxCeleb_ID")

#buscar coincidencias entre columna Name del dataframe age_test y df_vox_final
df_coincidencias7 = comparar_y_eliminar(age_test, df_vox_final, "Name", "Name")

#buscar coincidencias entre columna video_id del dataframe age_test y la columna VoxCeleb_ID df_audios
df_coincidencias8 = comparar_y_eliminar(age_test, df_audios, "video_id", "VoxCeleb_ID")

#buscar coincidencias entre columna video_id del dataframe age_test y la columna VoxCeleb_ID df_audios
df_coincidencias9 = comparar_y_eliminar(dataframe_filtrado, df_audios, "video_id", "VoxCeleb_ID")


# Contar cuántos elementos son femeninos
num_femeninos = len(df_vox1_final[df_vox1_final['Genero'] == 'f'])

# Contar cuántos elementos son masculinos
num_masculinos = len(df_vox1_final[df_vox1_final['Genero'] == 'm'])