# Sistema de Clasificación de Género por Voz usando Deep Learning 🎤

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)
![Librosa](https://img.shields.io/badge/Librosa-0.8+-green.svg)

Sistema de clasificación automática del género del hablante utilizando características acústicas y redes neuronales convolucionales. Este proyecto forma parte de una tesis de maestría que explora diferentes técnicas de procesamiento de señales de audio y aprendizaje profundo.

## 🔍 Palabras Clave

`reconocimiento-de-voz` `deep-learning` `procesamiento-de-audio` `clasificacion-de-genero` `pytorch` `cnn` `redes-neuronales` `espectrogramas` `mfcc` `mel-spectrograms` `voxceleb` `procesamiento-de-señales` `machine-learning` `python` `librosa` `audio-processing` `gender-recognition` `voice-processing` `tesis` `audio-features` `speaker-recognition`

## 🚀 Características Principales

- Procesamiento avanzado de señales de audio
- Extracción de múltiples características acústicas:
  - Espectrogramas
  - Espectrogramas Mel
  - Coeficientes MFCC
- Modelo CNN personalizado para clasificación
- Pipeline completo de preprocesamiento y entrenamiento
- Soporte para el dataset VoxCeleb

## 📊 Estructura del Proyecto

```
├── audio preprocessing/     # Preprocesamiento de señales de audio
├── dataset/                # Manejo y organización del dataset
├── extract features/       # Extracción de características
├── train test/            # Entrenamiento y evaluación
└── docs/                  # Documentación
```

## 🛠️ Características Técnicas

### Procesamiento de Audio
- Frecuencia de muestreo: 16kHz
- Ventana FFT: 1024 muestras
- Ventana de análisis: Hann
- Hop length: 160 muestras
- Bandas Mel: 224

### Arquitectura CNN
- Capa convolucional (1->16 canales)
- ReLU
- MaxPooling
- Fully connected (16*32*32 -> 128)
- Fully connected (128 -> num_classes)

## 🔧 Requisitos

```
python >= 3.7
torch >= 1.7
librosa >= 0.8
numpy
pandas
matplotlib
pillow
tqdm
```

## 📥 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/tesis.git
cd tesis
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🚦 Uso
-Se puede aplicar a aplicaciones en dispositivos de bajo rendimiento computacional, ya que ejecutas modelos como Mobilnet.

### Preprocesamiento de Audio
```bash
python audio_preprocessing/procesamiento_audio.py
```

### Extracción de Características
```bash
python extract_features/Guardar_MelSpectograma.py
python extract_features/Guardar_MFCC.py
python extract_features/Guardar_Spectograma.py
```

### Entrenamiento del Modelo
```bash
python train_test/cnn_pytorch.py
```

## 📊 Estructura del Dataset

```
dataset/
├── train/
│   ├── male/    # ~56,000 muestras
│   └── female/  # ~56,000 muestras
└── test/
    ├── male/    # ~5,500 muestras
    └── female/  # ~5,500 muestras
```

## 📈 Resultados

- Dataset: VoxCeleb
- Muestras totales: ~123,000
- División: 90% entrenamiento, 10% prueba
- Balanceado por género
- Accuracy: 98.95%

## 👥 Autores

- MSc. Jorge Luis Jorrin
- Asesores:
  - Dra. Mariko Nakano

## 📄 Licencia

Este proyecto está bajo la Licencia GNU GENERAL PUBLIC LICENSE - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🔗 Referencias

- [VoxCeleb Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- Enlaces a papers relevantes
