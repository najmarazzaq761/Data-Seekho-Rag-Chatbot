import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setting page configuration 
st.set_page_config(page_title="✨ Data Seekho Guide", page_icon="🧠", layout="wide")
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("Welcome to the Data Seekho Guide developed by Najma Razzaq. This app is designed to provide you with any information about Data Seekho.")
st.markdown("<h1 style='text-align: center;'><span style='color: #7abd06;'>Data</span> <span style='color: white;'>Seekho Guide</span></h1>", unsafe_allow_html=True)

# Function to fetch all internal links from the website
@st.cache_data
def fetch_all_links(base_url):
    """
    Fetch all internal links from the website's base URL.
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/"):  # Internal links
            links.add(base_url + href.strip("/"))
        elif base_url in href: 
            links.add(href)
    
    return list(links)

# Load data from all pages of the website
@st.cache_data
def load_data():
    """
    Load data from all pages of the website.
    """
    base_url = "https://dataseekho.com/"
    all_links = fetch_all_links(base_url)

    # Use WebBaseLoader to load data from all links
    all_data = []
    for link in all_links:
        try:
            loader = WebBaseLoader([link])
            all_data.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load data from {link}: {e}")
    
    return all_data

# Split the loaded data into chunks
@st.cache_data
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    return text_splitter.split_documents(_data)

# Create a vector store using FAISS
@st.cache_resource
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process data
data = load_data()
docs = split_data(data)
vectorstore = create_vector_store(docs)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0, max_tokens=None, timeout=None)

# Defining the prompt template
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
if st.button("submit"):
       question_answer_chain = create_stuff_documents_chain(llm, prompt)
       rag_chain = create_retrieval_chain(retriever, question_answer_chain)
       response = rag_chain.invoke({"input": query})
       st.write(response["answer"])
