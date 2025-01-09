# Librerías y Recursos
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from st_social_media_links import SocialMediaIcons
import numpy as np
import os

# Configuración de la página de Streamlit
st.set_page_config(page_title="Consultas Inteligentes PDF", page_icon=":books:", layout="wide")

# Estilo CSS personalizado
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2E2E2E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.markdown("<h1 style='text-align: center; color: white;'>Consultas Inteligentes PDF</h1>", unsafe_allow_html=True)
st.image("https://cdn-kktxrz66sku8.vultrcdn.com/wp-content/uploads/2023/02/Save-ChatGPT-Conversations-as-a-PDF.jpg", width=100)

# Entrada de clave API y archivo PDF
openai_api_key = st.text_input("Introduce tu API Key de OpenAI", type="password")
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

# Función para procesar el PDF
@st.cache_data
def process_pdf(file):
    pdf_reader = PdfReader(file)
    return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

# Función para crear y cargar embeddings
@st.cache_resource
def create_vector_store(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(docs, embeddings_model)
    return vector_store

# Procesamiento del archivo PDF y generación de embeddings
if uploaded_file and openai_api_key:
    try:
        # Leer y procesar el archivo PDF
        with st.spinner("Procesando el archivo PDF..."):
            text = process_pdf(uploaded_file)

        if not text:
            st.error("El archivo PDF no contiene texto legible.")
        else:
            # Dividir el texto en chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = text_splitter.split_text(text)

            # Crear el almacén de vectores
            with st.spinner("Generando embeddings y configurando búsqueda..."):
                vector_store = create_vector_store(chunks)

            # Entrada de pregunta al documento PDF
            pregunta = st.text_input("Haz una pregunta al PDF")

            if pregunta:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                llm = ChatOpenAI(model_name="gpt-3.5-turbo")
                qa_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=vector_store, chain_type="stuff")

                # Realizar la consulta
                with st.spinner("Generando respuesta..."):
                    respuesta = qa_chain.run(pregunta)
                
                st.markdown("<h3 style='color: white;'>Respuesta:</h3>", unsafe_allow_html=True)
                st.write(respuesta)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")
else:
    st.warning("Por favor, introduce tu API Key de OpenAI y sube un archivo PDF para continuar.")

# Pie de página con información del desarrollador y enlaces sociales
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
