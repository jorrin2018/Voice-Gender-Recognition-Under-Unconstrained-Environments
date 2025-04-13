# -*- coding: utf-8 -*-
"""
Script de preprocesamiento de datos para el dataset VoxCeleb
Este script realiza tareas de preprocesamiento en los datos del dataset,
incluyendo lectura de archivos CSV, extracción de campos específicos y
filtrado de datos basado en criterios predefinidos.

Funcionalidades:
- Lectura y procesamiento de archivos CSV
- Extracción del primer campo de cada archivo
- Comparación y filtrado de DataFrames
- Manejo de metadatos extendidos

@author: MCIM1
"""

# Importación de librerías necesarias
import os  # Para operaciones de sistema de archivos
import pandas as pd  # Para manejo de datos estructurados

# ============================
# Configuración inicial
# ============================

# Ruta del directorio que contiene los archivos CSV
directorio = 'D:/JORRIN/DATASET/VoxCeleb'

# Carga del archivo de metadata extendido
final_dataframe_extended = pd.read_csv("final_dataframe_extended.csv")
df_nombres = final_dataframe_extended["description"]

# ============================
# Funciones auxiliares
# ============================

def leer_csv_obtener_primer_campo(ruta_csv):
    """
    Lee un archivo CSV y extrae el valor del primer campo de la primera fila.
    
    Args:
        ruta_csv (str): Ruta al archivo CSV a procesar
    
    Returns:
        any: Valor del primer campo o None si el archivo está vacío
    """
    df = pd.read_csv(ruta_csv)
    primer_campo = df.iloc[0, 0] if not df.empty else None
    return primer_campo

def leer_carpetas_obtener_primer_campo(directorio):
    """
    Recorre todas las carpetas y subcarpetas buscando archivos CSV
    y extrae el primer campo de cada uno.
    
    Args:
        directorio (str): Ruta del directorio raíz a explorar
    
    Returns:
        DataFrame: DataFrame con los primeros campos encontrados
    """
    datos = []
    
    for carpeta, _, archivos in os.walk(directorio):
        for archivo in archivos:
            if archivo.endswith('.csv'):
                ruta_csv = os.path.join(carpeta, archivo)
                primer_campo = leer_csv_obtener_primer_campo(ruta_csv)
                if primer_campo:
                    datos.append(primer_campo)
    
    return pd.DataFrame(datos, columns=['PrimerCampo'])

def comparar_y_eliminar(df1, df2, columna):
    """
    Filtra un DataFrame basándose en los valores coincidentes de una columna
    entre dos DataFrames.
    
    Args:
        df1 (DataFrame): Primer DataFrame a comparar
        df2 (DataFrame): Segundo DataFrame a comparar
        columna (str): Nombre de la columna para comparar
    
    Returns:
        DataFrame: DataFrame filtrado con las filas coincidentes
    """
    df1_filtrado = df1[df1[columna].isin(df2[columna])]
    return df1_filtrado

# ============================
# Ejecución principal
# ============================

# Obtener DataFrame con los primeros campos
dataframe = leer_carpetas_obtener_primer_campo(directorio)

# Renombrar la columna para coincidencia
dataframe.rename(columns={'PrimerCampo': 'video_id'}, inplace=True)

# Comparar y filtrar DataFrames
df1 = final_dataframe_extended
df2 = dataframe
columna = 'video_id'

# Obtener DataFrame filtrado final
df1_filtrado = comparar_y_eliminar(df1, df2, columna)

# ============================
# Resultados
# ============================

# El DataFrame filtrado está almacenado en la variable `df1_filtrado`
# Puedes guardar o procesar este DataFrame según sea necesario.






