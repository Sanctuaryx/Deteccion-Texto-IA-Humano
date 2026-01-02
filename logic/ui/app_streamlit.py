import requests
import streamlit as st

st.set_page_config(
    page_title="Detector IA vs. Humano (ES)",
    layout="centered",
)

st.title("Detector de texto generado por IA (castellano)")

api_url = st.text_input(
    "URL de la API",
    value="http://localhost:8000/predict",
)

text = st.text_area(
    "Pega tu texto aquí",
    height=260,
    placeholder="Escribe o pega un texto en español...",
)

if st.button("Evaluar", use_container_width=True):
    if not text.strip():
        st.warning("Introduce un texto primero.")
    else:
        try:
            resp = requests.post(api_url, json={"text": text}, timeout=60)
            st.write("Status:", resp.status_code)
            st.json(resp.json())
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")

st.caption(
    "Nota: textos muy cortos se consideran 'indeterminados'. "
    "El modelo debe usarse como herramienta de apoyo, no como única base para sanciones."
)
