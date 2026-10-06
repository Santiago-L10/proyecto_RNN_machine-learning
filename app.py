import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import os
from deep_translator import MyMemoryTranslator

# Configuración visual de la página
st.set_page_config(
    page_title="Análisis de Sentimientos - LSTM",
    page_icon="🎬",
    layout="centered"
)

# 1. Carga optimizada de modelo y vocabulario
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "modelo_sentimientos_lstm.keras")
    vocab_path = os.path.join("models", "word_index.json")
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        st.error("No se encontraron los artefactos en la carpeta 'models/'. Verifica las rutas.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        word_index = json.load(f)
        
    return model, word_index

model, word_index = load_artifacts()

# 2. Función de preprocesamiento de texto
def preprocess_text(text, vocab, max_len=200):
    clean_text = text.lower().replace('<br />', ' ').replace('.', ' ').replace(',', ' ')
    tokens = clean_text.split()
    
    encoded = []
    for token in tokens:
        idx = vocab.get(token)
        # Desfase de +3 por convención interna del dataset IMDb
        if idx is not None and (idx + 3) < 10000:
            encoded.append(idx + 3)
        else:
            encoded.append(2)  # Token 'out-of-vocabulary'
            
    padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')
    return padded

# 3. Interfaz de usuario
st.title("🎬 Clasificador de Sentimientos en Reseñas")
st.markdown("""
Esta aplicación utiliza una red neuronal **Bidirectional LSTM** entrenada 
sobre **IMDb (50.000 reseñas)**. Admite opiniones en **español o inglés** mediante traducción automática previa a la inferencia.
""")

st.write("---")

# Ejemplos rápidos predefinidos
ejemplos = {
    "Seleccionar un ejemplo...": "",
    "Reseña Positiva (EN)": "This movie was an absolute masterpiece with brilliant acting and stunning visuals.",
    "Reseña Positiva (ES)": "Una experiencia cinematográfica inolvidable, actuaciones impecables y música hermosa.",
    "Reseña Negativa (EN)": "A total waste of time. The plot was completely incoherent and boring.",
    "Reseña Negativa (ES)": "La peor película que he visto en mucho tiempo, dirección espantosa y diálogos ridículos."
}

seleccion = st.selectbox("Prueba con un ejemplo predefinido o escribe abajo:", list(ejemplos.keys()))
default_text = ejemplos[seleccion]

user_input = st.text_area(
    "Escribe o pega una reseña (en español o inglés):",
    value=default_text,
    height=140,
    placeholder="Ej: La película fue una completa obra de arte, la actuación y la música fueron increíbles..."
)

# Botón de análisis
if st.button("Analizar Sentimiento", type="primary"):
    if not user_input.strip():
        st.warning("Por favor, ingresa una reseña antes de realizar la predicción.")
    else:
        with st.spinner("Procesando traducción e inferencia..."):
            texto_ingles = user_input.strip()
            
            # Traducción con MyMemory para evitar bloqueos por límite de peticiones
            try:
                traduccion = MyMemoryTranslator(source='es-ES', target='en-GB').translate(user_input)
                if traduccion and len(traduccion.strip()) > 0:
                    texto_ingles = traduccion
            except Exception:
                texto_ingles = user_input

            sequence = preprocess_text(texto_ingles, word_index)
            raw_prob = float(model.predict(sequence, verbose=0)[0][0])
            
            is_positive = raw_prob >= 0.5
            confidence = raw_prob if is_positive else (1.0 - raw_prob)
            
        st.write("### Resultado del Análisis")
        col1, col2 = st.columns(2)
        
        with col1:
            if is_positive:
                st.success("### Sentimiento: **POSITIVO** 😊")
            else:
                st.error("### Sentimiento: **NEGATIVO** 😞")
                
        with col2:
            st.metric(label="Confianza del Modelo", value=f"{confidence * 100:.2f}%")
            
        st.progress(float(confidence))
        
        with st.expander("Ver texto procesado por el modelo (en inglés)"):
            st.write(f"**Traducción utilizada:** {texto_ingles}")
            st.caption(f"Probabilidad sigmoide cruda: `{raw_prob:.4f}` (Valores ≥ 0.5 indican sentimiento positivo).")
            
st.write("---")
st.markdown("""
**Detalles del Modelo:**
* **Arquitectura:** Embedding (64d) $\\rightarrow$ Bidirectional LSTM (64 unidades) $\\rightarrow$ Dropout $\\rightarrow$ Densa (32) $\\rightarrow$ Sigmoide.
* **Vocabulario:** 10.000 palabras más frecuentes. Longitud máxima por reseña: 200 tokens.
""")