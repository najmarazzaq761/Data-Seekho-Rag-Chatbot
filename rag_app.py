# importing necesary libraries
import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import create_rag_chain,get_response,load_previous_chat
from dotenv import load_dotenv
load_dotenv()

# set the session variable
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Setting page configuration 
st.set_page_config(page_title="✨ Data Seekho Guide", page_icon="🧠", layout="wide")

# Sidebar content 
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNl6Gok8ubOtQLNgMDmKQQGFdV5OtfJWYSOqyYTfM-uNml-vaBpavqlUXpdYdoHWed0LY&usqp=CAU", use_column_width=True)
st.sidebar.markdown("Welcome to the Data Seekho Guide developed by Najma Razzaq. This app is designed to provide you with information about Data Seekho.")

# Main titles
st.header("✨ Data Seekho Guide")

# Load data from a website
loader = WebBaseLoader([
    "https://dataseekho.com/",
    "https://dataseekho.com/free-courses/",
    "https://dataseekho.com/join-us/",
    "https://dataseekho.com/about-us/"
])
data = loader.load()

# Spliting  data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Loading the Google API key
google_api_key = st.secrets["GOOGLE_API_KEY"]
# genai.configure(google_api_key=google_api_key)

# Initializing embeddings and vector store using FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

# Seting up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key, temperature=0, max_tokens=None, timeout=None)

# create the RAG chain
rag_chain=create_rag_chain(llm=llm,retriever=retriever)

# load the previous chat
with st.container():
    load_previous_chat()

# User input and response generation
query = st.chat_input(placeholder="🗣️ Enter your query:")

if query:
    st.write(f'💬 You: {query}') 
    response = get_response(rag_chain,query)
    st.write(f'🤖 Bot: {response}')