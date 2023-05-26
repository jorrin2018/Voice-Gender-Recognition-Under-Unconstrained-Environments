# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:20:40 2023

@author: MCIM1
"""

import os
import pandas as pd
import csv
# def leer_carpetas_directorio(directorio):
#     archivos_csv = []
    
#     # Recorre todas las carpetas y archivos del directorio
#     for carpeta, subcarpetas, archivos in os.walk(directorio):
#         for archivo in archivos:
#             if archivo.endswith('.csv'):  # Verifica que el archivo sea de tipo CSV
#                 data = pd.read_csv(archivo)       
#                 primera_fila = data.loc[0,"Video_id"] 
#                 archivos_csv.append(primera_fila)
    
#     # Lee los archivos CSV y guarda los nombres en un DataFrame
#     nombres_df = pd.DataFrame(columns=['Nombre'])
#     for csv in archivos_csv:
       
#         nombres_df = nombres_df.append({'Nombre': csv}, ignore_index=True)
    
#     return nombres_df


# # Ejemplo de uso
directorio = 'D:/JORRIN/DATASET/VoxCeleb'  # Reemplaza con la ruta correcta de tu directorio
# nombres_csv_df = leer_carpetas_directorio(directorio)

final_dataframe_extended=pd.read_csv("final_dataframe_extended.csv")
df_nombres = final_dataframe_extended["description"]


def leer_csv_obtener_primer_campo(ruta_csv):
    df = pd.read_csv(ruta_csv)
    primer_campo = df.iloc[0, 0] if not df.empty else None
    return primer_campo
    

def leer_carpetas_obtener_primer_campo(directorio):
    datos = []
    
    for carpeta, subcarpetas, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith('.csv'):
                ruta_csv = os.path.join(carpeta, archivo)
                primer_campo = leer_csv_obtener_primer_campo(ruta_csv)
                if primer_campo:
                    datos.append(primer_campo)
    
    df = pd.DataFrame(datos, columns=['PrimerCampo'])
    return df

# Ejemplo de uso

dataframe = leer_carpetas_obtener_primer_campo(directorio)

dataframe1 = dataframe

dataframe.rename(columns={'PrimerCampo': 'video_id'}, inplace=True)




def comparar_y_eliminar(df1, df2, columna):
    df1_filtrado = df1[df1[columna].isin(df2[columna])]
    return df1_filtrado

# Ejemplo de uso
df1 = final_dataframe_extended
df2 = dataframe1

columna = 'video_id'  # Columna a comparar

df1_filtrado = comparar_y_eliminar(df1, df2, columna)






