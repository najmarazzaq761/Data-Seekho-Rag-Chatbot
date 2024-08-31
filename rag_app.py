import streamlit as st
import os
import google.generativeai as genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set page configuration with a custom title and icon
st.set_page_config(page_title="✨ Data Seekho Guide", page_icon="🧠", layout="wide")

# Sidebar content with Data Seekho logo
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNl6Gok8ubOtQLNgMDmKQQGFdV5OtfJWYSOqyYTfM-uNml-vaBpavqlUXpdYdoHWed0LY&usqp=CAU", use_column_width=True)
# st.sidebar.markdown("# Data Seekho Guide")
st.sidebar.markdown("Welcome to the Data Seekho Guide developed by Najma Razzaq. This app is designed to provide you any information about data seekho.")

# Main title with custom style
st.markdown("<h1 style='text-align: center; color: #FF6347;'>✨ Data Seekho Guide</h1>", unsafe_allow_html=True)

# Load data from a website
loader = WebBaseLoader(["https://dataseekho.com/","https://dataseekho.com/free-courses/","https://dataseekho.com/join-us/","https://www.f6s.com/company/dataseekho#about", "https://dataseekho.com/about-us/"])
data = loader.load()

# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Fetch the API key from the environment
google_api_key=os.environ["GOOGLE_API_KEY"] = "AIzaSyD4BNwGnPBR-by6aUieAhdBHyAzRXzfStc"

# Check if the API key was found
if google_api_key is None:
    raise ValueError("API_KEY environment variable not found. Please set it.")

# Initialize embeddings and vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key, temperature=0, max_tokens=None, timeout=None)

# Define the prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# User input and response generation
query = st.text_input("🗣️ Enter your query:")

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
