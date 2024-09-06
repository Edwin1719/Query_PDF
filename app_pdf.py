# Librerias y Recursos
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

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

if uploaded_file and openai_api_key:
    # Leer el archivo PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Crear chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Definición del modelo y creación del Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Preguntar al documento PDF
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")

    pregunta = st.text_input("Haz una pregunta al PDF")
    if pregunta:
        docs = knowledge_base.similarity_search(pregunta, 4)
        respuesta = chain.run(input_documents=docs, question=pregunta)
        st.markdown("<h3 style='color: white;'>Respuesta:</h3>", unsafe_allow_html=True)
        st.write(f"{respuesta}")

else:
    st.warning("Por favor, introduce tu API Key de OpenAI y sube un archivo PDF para continuar.")

# Pie de página
st.markdown(
    """
    <footer style='text-align: center; color: white;'>
        <p>Desarrollado por Edwin Quintero Alzate | Sígueme en mis redes sociales:
        <a href='https://www.linkedin.com/in/edwinquintero0329/' style='color: white;'>LinkedIn</a>,
        <a href='https://github.com/Edwin1719' style='color: white;'>GitHub</a></p>
    </footer>
    """,
    unsafe_allow_html=True)
