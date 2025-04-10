# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:20:40 2023

@author: MCIM1
"""

# Importación de librerías necesarias
import os
import pandas as pd

# ============================
# Configuración inicial
# ============================

# Ruta del directorio que contiene los archivos CSV
directorio = 'D:/JORRIN/DATASET/VoxCeleb'  # Reemplaza con la ruta correcta de tu directorio

# Carga del archivo de metadata
final_dataframe_extended = pd.read_csv("final_dataframe_extended.csv")
df_nombres = final_dataframe_extended["description"]

# ============================
# Funciones auxiliares
# ============================

def leer_csv_obtener_primer_campo(ruta_csv):
    """
    Lee un archivo CSV y devuelve el valor del primer campo de la primera fila.
    Si el archivo está vacío, devuelve None.
    """
    df = pd.read_csv(ruta_csv)
    primer_campo = df.iloc[0, 0] if not df.empty else None
    return primer_campo

def leer_carpetas_obtener_primer_campo(directorio):
    """
    Recorre todas las carpetas y archivos de un directorio, busca archivos CSV,
    y extrae el primer campo de cada archivo encontrado.
    """
    datos = []
    
    for carpeta, subcarpetas, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith('.csv'):  # Verifica que el archivo sea de tipo CSV
                ruta_csv = os.path.join(carpeta, archivo)
                primer_campo = leer_csv_obtener_primer_campo(ruta_csv)
                if primer_campo:
                    datos.append(primer_campo)
    
    # Crea un DataFrame con los datos obtenidos
    df = pd.DataFrame(datos, columns=['PrimerCampo'])
    return df

def comparar_y_eliminar(df1, df2, columna):
    """
    Compara dos DataFrames en base a una columna específica y devuelve un nuevo
    DataFrame con las filas de df1 que coinciden con los valores de df2.
    """
    df1_filtrado = df1[df1[columna].isin(df2[columna])]
    return df1_filtrado

# ============================
# Ejecución principal
# ============================

# Obtención de los datos del directorio
dataframe = leer_carpetas_obtener_primer_campo(directorio)

# Renombrar la columna para que coincida con la columna de comparación
dataframe.rename(columns={'PrimerCampo': 'video_id'}, inplace=True)

# Comparar y filtrar los DataFrames
df1 = final_dataframe_extended
df2 = dataframe  # DataFrame obtenido del directorio
columna = 'video_id'  # Columna a comparar

# Filtrar el DataFrame en base a la comparación
df1_filtrado = comparar_y_eliminar(df1, df2, columna)

# ============================
# Resultados
# ============================

# El DataFrame filtrado está almacenado en la variable `df1_filtrado`
# Puedes guardar o procesar este DataFrame según sea necesario.






