# Librerías y Recursos
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
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

# Título de la aplicación
st.markdown("<h1 style='text-align: center; color: white;'>Consultas Inteligentes PDF</h1>", unsafe_allow_html=True)
st.image("https://cdn-kktxrz66sku8.vultrcdn.com/wp-content/uploads/2023/02/Save-ChatGPT-Conversations-as-a-PDF.jpg", width=100)

# Cargar la API de OpenAI
openai_api_key = st.text_input("Introduce tu API Key de OpenAI", type="password")

# Cargar el archivo PDF
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

# Función para procesar el PDF
@st.cache_data
def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Función para realizar la búsqueda de similitud con sklearn
def similarity_search_sklearn(question, embeddings, chunks, k=3):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = np.array(embeddings_model.embed_query(question)).reshape(1, -1)
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    most_similar_indices = similarities.argsort()[-k:][::-1]  # Obtener los índices de los k más similares
    return [chunks[i] for i in most_similar_indices]

# Procesamiento del archivo PDF y generación de embeddings
if uploaded_file and openai_api_key:
    # Leer el archivo PDF
    text = process_pdf(uploaded_file)

    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)

    # Crear embeddings usando HuggingFace
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array([np.array(embeddings_model.embed_documents([chunk])[0]) for chunk in chunks])

    # Preguntar al documento PDF
    pregunta = st.text_input("Haz una pregunta al PDF")
    
    if pregunta:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")

        # Buscar los chunks más relevantes con cosine similarity
        docs = similarity_search_sklearn(pregunta, embeddings, chunks)

        # Convertir los chunks en objetos Document
        docs = [Document(page_content=chunk) for chunk in docs]

        # Ejecutar la cadena de preguntas y respuestas
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