# Reconocimiento de Patrones en Imágenes

Código desarrollado para el curso de **Reconocimiento de Patrones en Imágenes**. El repositorio reúne scripts cortos en Python, organizados por clase, que cubren el flujo completo de un sistema de reconocimiento: lectura y manipulación de imágenes, extracción y selección de características, clasificación, evaluación de modelos y agrupamiento (clustering).

## Contenido del curso

Cada script sigue la convención `clase_XX_YY_tema.py`, donde `XX` es el número de clase e `YY` el orden sugerido dentro de ella.

| Clase | Tema | Contenidos principales |
|:-----:|------|------------------------|
| 01 | Fundamentos de imágenes digitales | Lectura de imágenes, conversión BGR→RGB, separación de canales, histogramas, valores de píxel, regiones de interés (ROI), cambio de escala |
| 02 | Descriptores de forma | Centro de masa, perímetro, momentos, contornos, descriptores de Fourier, ajuste de elipses, distancias y transformada de distancia, anexos de FFT en señales y audio |
| 03 | Filtros, texturas y puntos de interés | Promedio, derivadas, filtro gaussiano, laplaciano, contraste, matrices de co-ocurrencia (GLCM), correlación, momentos de Hu, detectores Harris, SURF, SIFT y KAZE, *matching* y visión estéreo |
| 04 | Reducción de dimensionalidad y cuantización | PCA (datos e imágenes), cuantización vectorial (VQ) en una y varias dimensiones, VQ con SciPy |
| 05 | Selección de características | Criterio de Fisher, proyección LDA, SBS y SFS, *branch and bound*, optimización con Optuna, LASSO (con *grid search*), Random Forest |
| 06 | Clasificadores | Normalización, KNN, distancia de Mahalanobis, LDA, árboles de decisión, Naive Bayes, redes neuronales y *backpropagation* |
| 07 | Evaluación de modelos | Validación cruzada *k-fold*, *bootstrap*, clases desbalanceadas con SMOTE, curvas ROC/AUC, comparación de modelos |
| 08 | Aprendizaje no supervisado | K-means, DBSCAN, clustering jerárquico (dendrogramas), Fuzzy C-means, estadístico de Hopkins |

## Estructura del repositorio

```
.
├── clase_01_*.py … clase_08_*.py   # Scripts del curso, uno por tema
├── data/
│   ├── imagenes/                   # Imágenes de prueba (lena, cameraman, rice, texturas, …)
│   ├── texturas_fisher/            # Texturas usadas en selección de características
│   ├── estereo/                    # Pares de fotografías para matching estéreo
│   ├── model/                      # Modelo PyTorch preentrenado (modelo_4.pth)
│   └── *.csv                       # Conjuntos de datos tabulares de los ejemplos
├── tree.dot / tree.png             # Árbol de decisión exportado (clase 06)
└── LICENSE
```

## Requisitos

- Python 3.9 o superior
- Bibliotecas principales:

```bash
pip install numpy scipy matplotlib pandas seaborn opencv-contrib-python scikit-image scikit-learn imutils
```

- Bibliotecas usadas solo en algunos scripts:

| Paquete | Usado en |
|---------|----------|
| `optuna` | Optimización de hiperparámetros (clases 05 y 06) |
| `imbalanced-learn` | SMOTE (clase 07) |
| `torch` | Red neuronal (`clase_06_15_RNN_Full.py`) |
| `tensorflow` / `keras` | Conjunto Fashion-MNIST (clase 06) |
| `pyclustertend` | Estadístico de Hopkins (`clase_08_09_test_hopkins.py`) |
| `pydub` | Lectura de audio (`clase_02_13_audio.py`, requiere `ffmpeg`) |
| `graphviz` | Exportar árboles de decisión a imagen |

> **Nota:** SURF está patentado y solo está disponible en compilaciones de OpenCV con los módulos *non-free* habilitados. Si `clase_03_16_SURF.py` falla, usa SIFT o KAZE como alternativa.

## Uso

Clona el repositorio y ejecuta los scripts **desde la raíz del proyecto**, ya que las rutas a los datos son relativas (por ejemplo, `data/imagenes/lena.png`):

```bash
git clone https://github.com/mlacarrasco/image_pattern_recognition.git
cd image_pattern_recognition
python clase_01_01_lectura.py
```

Algunos conjuntos de imágenes también se incluyen comprimidos (`data/imagenes.zip`, `data/texturas_fisher.zip`, `data/imagenes/texturas_1_9.zip`) para facilitar su descarga.

## Licencia

Distribuido bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
