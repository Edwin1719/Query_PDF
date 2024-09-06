# Librerias y Recursos
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from st_social_media_links import SocialMediaIcons
import asyncio

# Configuración de la página Streamlit
st.set_page_config(page_title="Consultas Inteligentes PDF", page_icon=":books:", layout="wide", initial_sidebar_state="expanded")

# Estilo CSS personalizado
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2E2E2E;
        color: white;
    }
    .stTitle {
        color: white;
    }
    .stTextInput > div > input {
        color: black;
    }
    .stTextArea > div > textarea {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicacion y logo
st.markdown("<h1 style='text-align: center; color: white;'>Consultas Inteligentes PDF</h1>", unsafe_allow_html=True)
st.image("https://cdn-kktxrz66sku8.vultrcdn.com/wp-content/uploads/2023/02/Save-ChatGPT-Conversations-as-a-PDF.jpg", width=100)

# Cargar la API de OpenAI
openai_api_key = st.text_input("Introduce tu API Key de OpenAI", type="password")

# Cargar el archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

@st.cache_data
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if uploaded_file and openai_api_key:
    # Leer el archivo PDF
    text = process_pdf(uploaded_file)

    # Crear chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Definición del modelo y creación del Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Preguntar al documento PDF
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")

    pregunta = st.text_input("Haz una pregunta al PDF")
    if pregunta:
        docs = knowledge_base.similarity_search(pregunta, 3)
        respuesta = chain.run(input_documents=docs, question=pregunta)
        st.markdown("<h3 style='color: white;'>Respuesta:</h3>", unsafe_allow_html=True)
        st.write(f"{respuesta}")

else:
    st.warning("Por favor, introduce tu API Key de OpenAI y sube un archivo PDF para continuar.")

# Pie de página con información del desarrollador y logos de redes sociales
st.markdown("""
---
**Desarrollador:** Edwin Quintero Alzate<br>
**Email:** egqa1975@gmail.com<br>
""")

social_media_links = [
    "https://www.facebook.com/edwin.quinteroalzate",
    "https://www.linkedin.com/in/edwinquintero0329/",
    "https://github.com/Edwin1719"]

social_media_icons = SocialMediaIcons(social_media_links)
social_media_icons.render()