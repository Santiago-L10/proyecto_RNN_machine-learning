# Clasificación de Sentimientos en Reseñas de Películas (IMDb) con Redes LSTM

Este repositorio contiene el desarrollo completo de un modelo de Deep Learning para análisis de sentimiento sobre el conjunto de datos **IMDb (50.000 reseñas)**, utilizando una arquitectura de red neuronal recurrente **Bidirectional LSTM**, junto con una aplicación web interactiva desarrollada en **Streamlit**.

---

## 📁 Estructura del Repositorio

```text
├── models/
│   ├── modelo_sentimientos_lstm.keras   # Modelo entrenado persistido
│   └── word_index.json                  # Mapeo del vocabulario procesado
├── notebook/
│   └── experimentacion_deep_learning.ipynb # Notebook original de Google Colab
├── app.py                               # Aplicación interactiva de Streamlit
├── requirements.txt                     # Dependencias del entorno
└── README.md                            # Documentación del proyecto
```

## 🧠 Arquitectura del Modelo y Entrenamiento

* **Dataset:** IMDb Reviews (25.000 para entrenamiento/validación y 25.000 para evaluación final).

* **Preprocesamiento:**

    * Límite de vocabulario: 10.000 palabras más frecuentes.

    * Longitud de secuencia (MAX_LEN): 200 tokens con relleno (padding='post').

    * División estratificada: 80% Entrenamiento (20.000), 20% Validación (5.000) y Test (25.000).

* **Capas de la Red:**

    1. Embedding(input_dim=10000, output_dim=64, input_length=200)

    2. Bidirectional(LSTM(64))

    3. Dropout(0.3)

    4. Dense(32, activation='relu')

    5. Dropout(0.3)

    6. Dense(1, activation='sigmoid')

* **Optimización y Regularización:** Optimizador Adam, pérdida binary_crossentropy, y EarlyStopping restaurando los mejores pesos de validación.

## 📊 Métricas Obtenidas en Prueba (Test Set: 25.000 muestras)

* **Accuracy:** 84.86%

* **Precision:** Negativa = 0.87 | Positiva = 0.83

* **Recall:** Negativa = 0.83 | Positiva = 0.87

* **F1-Score:** Negativa = 0.84 | Positiva = 0.85

## 🚀 Instrucciones de Ejecución Local

### 1. Clonar el repositorio y acceder a la carpeta

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd proyecto-deep-learning-lstm
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Lanzar la aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en http://localhost:8501.