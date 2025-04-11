# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:45:19 2023

@author: MCIM
"""
import numpy as np
import pandas as pd

def comparar_y_eliminar(df1, df2, columna):
    df1_filtrado = df1[df1[columna].isin(df2[columna])]
    return df1_filtrado

dataframe = pd.read_csv("E:/JORRIN/TESIS/DATASET/final_dataframe_extended.csv")
dataframe1 = pd.read_csv("E:/JORRIN/TESIS/DATASET/vox1_meta.csv")
age_train = pd.read_csv("E:/JORRIN/TESIS/DATASET/voxceleb_enrichment_age_gender-main/voxceleb_enrichment_age_gender-main/dataset/age-train.txt")

#extraer columna VoxCeleb ID del dataframe
lista_ID=[]

for i in list(range(1251)):
    texto = dataframe1.iat[i,0]
    texto1 = texto.split('\t')[0]
    lista_ID.append(texto1)
    
#filtrar dataframe para solo quedarme con las etiquetas que me interesan    
dataframe_filtrado = dataframe[["VoxCeleb_ID", "gender", "speaker_age_title_only"]]

# for i in list(range(dataframe.shape[0]+1)):
#     for j in lista_ID: 
#         if(dataframe_filtrado.iat[i,0] == j):
#             pass
#         else:
#             dataframe_filtrado=dataframe_filtrado.replace(dataframe_filtrado.iat[i,0],np.nan)

#convertir lista de IDS a dataframe 
df_ids=pd.DataFrame(lista_ID)
df_ids.columns=["VoxCeleb_ID"]

def comparar_y_eliminar(df1, df2, columna):
    df1_filtrado = df1[df1[columna].isin(df2[columna])]
    return df1_filtrado

# Ejemplo de uso
df1 = dataframe_filtrado
df2 = df_ids

columna = 'VoxCeleb_ID'  # Columna a comparar

df1_filtrado = comparar_y_eliminar(df1, df2, columna)


esta_presente = dataframe_filtrado["VoxCeleb_ID"].isin(["id10001"])

hay_true = esta_presente.any()

if hay_true:
    print("La serie contiene al menos un True.")
else:
    print("La serie no contiene ningún True.")

