# -*- coding: utf-8 -*-
"""
Script para el procesamiento de señales de audio
Este script realiza el análisis y visualización de archivos de audio WAV, generando:
- Audiogramas (forma de onda)
- Espectrogramas
- Espectrogramas de Mel
- Coeficientes MFCC

El script está diseñado para procesar audios de voces masculinas y femeninas
del dataset VoxCeleb.

@author: MCIM
"""

# Importación de librerías necesarias
import os  # Para manejo de archivos y directorios
import numpy as np  # Para operaciones numéricas
import librosa  # Para procesamiento de audio
import librosa.display  # Para visualización de características de audio
import matplotlib.pyplot as plt  # Para graficación

# Inicialización de variables globales
samplerate = []  # Lista para almacenar las frecuencias de muestreo
lista_elementos = []  # Lista para almacenar nombres de archivos
lista_ruta = []  # Lista para almacenar rutas de archivos

# Configuración de rutas y parámetros
carpeta_raiz = 'E:/JORRIN/TESIS/DATASET/vox1_dev_wav/wav/id1'

# IDs de VoxCeleb a analizar (10 diferentes hablantes)
lista=['0001','0006','0002','0014','0009','0007','0010','0013','0018','0026']

# Generar lista con rutas completas para cada ID
lista_carpeta_raiz=[]
for i in range(10):
    lista_carpeta_raiz.append(carpeta_raiz+lista[i])

# Función para recorrer carpetas y subcarpetas
def recorrer_carpeta(carpeta_raiz):
    lista_ruta.clear()
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
    return [lista_elementos, lista_ruta]

#ciclo para recorer las 5 ubicaciones de los ficheros de audio
lista_MFCC=[]
for i in range(10):
    # Llamar a la función para explorar la carpeta raíz y sus subcarpetas
    resultado=recorrer_carpeta(lista_carpeta_raiz[i])
 
    #pasar las ubicaciones de los ficheros audios         
    filename = resultado[1][0] 
    #calculo del sample rate y graficar
    sr = librosa.get_samplerate(filename) #sr=8000
    samplerate.append(sr)
    waveform, sr = librosa.load(filename,sr=sr)
    plt.figure()
    if i%2==0:
        plt.title('Audiograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Hombre')
    else:
        plt.title('Audiograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Mujer')
    plt.plot(waveform)
    plt.show()
    
    #calculo del espectograma
    n_fft = 1024
    win_length = n_fft  # predeterminado 
    hop_length = 22
    window = "hann"
    S = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length  )
    MS = np.abs(S)**2
    Log_MS = librosa.power_to_db(MS)
    librosa.display.specshow(Log_MS, sr=sr, hop_length=hop_length, n_fft = n_fft, x_axis='s', y_axis='hz', cmap='viridis')
    
    # Agregar un título y etiquetas a los ejes
    if i%2==0:
        plt.title('Espectograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Hombre')                                                                                                                                                                 
    else:
        plt.title('Espectograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Mujer')
    plt.colorbar(format='%+2.0f dB')  # Agrega una barra de color
    
    # Mostrar la gráfica
    plt.show()
    
    
    #calcular melspectograma
    spect = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels =224 )
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    
    
    # Mostrar el espectrograma Mel utilizando librosa.display.specshow
    librosa.display.specshow(mel_spect, y_axis='mel', x_axis='time')
    
    # Agregar un título y etiquetas a los ejes
    if i%2==0:
        plt.title('Melspectograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Hombre')
    else:
        plt.title('Melspectograma' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Mujer')
    plt.colorbar(format='%+2.0f dB')  # Agrega una barra de color
    
    # Mostrar la gráfica
    plt.show()
    
    #calcular mfcc
    mfcc = librosa.feature.mfcc(y=waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr, 
    n_mfcc=224, window = window , dct_type=2)
    lista_MFCC.append(mfcc)
    
    # Mostrar los MFCC utilizando librosa.display.specshow
    librosa.display.specshow(mfcc, x_axis='time')
    
    # Agregar un título y etiquetas a los ejes
    if i%2==0:
        plt.title('MFCC' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Hombre')
    else:
        plt.title('MFCC' + ' '+ 'audio'+' ' + lista_carpeta_raiz[i][-7:] + ' '+'Clase: Mujer')
    plt.colorbar(format='%+2.0f dB')  # Agrega una barra de color
    
    # Mostrar la gráfica
    plt.show()





