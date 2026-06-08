import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUESTION_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/questions"


st.set_page_config(
    page_title="RAG Challenge Chat",
    layout="centered",
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "streamlit_user"


def ask_rag(question: str, user_name: str) -> str:
    """Envía una pregunta a la API RAG y devuelve la respuesta."""
    response = requests.post(
        QUESTION_ENDPOINT,
        json={
            "user_name": user_name,
            "question": question,
        },
        timeout=120,
    )
    response.raise_for_status()
    return str(response.json()["answer"])


with st.sidebar:
    st.title("Configuración de usuario")
    st.session_state.user_name = st.text_input(
        "Usuario",
        value=st.session_state.user_name,
    ).strip() or "streamlit_user"

    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Asistente RAG")

for message in st.session_state.messages: # los mensajes los guardo en st messages para resolverlo e manera sencilla, ya que actualmente no persisto el historial de la conversación realmente en bd y en el back los persisto en memoria por simplicidad
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribí tu pregunta sobre el documento"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        with st.spinner("Consultando el RAG..."):
            try:
                answer = ask_rag(
                    question=prompt,
                    user_name=st.session_state.user_name,
                )
                answer_placeholder.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except requests.RequestException as error:
                message = f"No se pudo obtener respuesta de la API: {error}"
                answer_placeholder.error(message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": message}
                )
