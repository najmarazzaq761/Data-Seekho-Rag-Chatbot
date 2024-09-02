from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
import os

def create_rag_chain(llm,retriever):
    try:
        # make a sytem message
        system_message = (
            "You are given a chat history and a new user question. "
            "Your task is to review both the chat history and the new question. "
            "If the new question references context from the chat history, reformulate it "
            "to ensure it is standalone and comprehensible. If the new question does not "
            "need any modification and is clear on its own, return it as is. "
            "Do not answer the question—focus solely on reformulating it for clarity if necessary."
        )

        # cousomize the prompt
        coustomize_system_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_message),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # set the history aware retriever
        history_aware_retriever=create_history_aware_retriever(
            llm=llm,retriever=retriever,prompt=coustomize_system_prompt
        )

        # make a question_answer prompt
        q_a_prompt = (
            "You are an expert assistant for  questions answering tasks "
            "Please use the following context to formulate your response to the user’s query. "
            "If the provided context does not allow you to answer the question, respond with 'I don't know.' "
            "Format your response using bullet points for clarity when detailing specific symptoms or key information. "
            "Otherwise, keep your response structured and engaging without using bullet points. "
            "Make sure your answer is clear, concise, and directly addresses the user's question in an attractive and informative manner.\n\n"
            "{context}"
        )


        # coustoize the Q&A prompt
        final_qa_prompt=ChatPromptTemplate(
            [
                ("system",q_a_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # create a document stuff
        question_answer_chain=create_stuff_documents_chain(llm,final_qa_prompt)

        # make the rag chain
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        return rag_chain
    except Exception as e:
        return str(e)

    
def get_response(rag_chain,query):
    """
    This fun is responsible for continuing the chat
    
    Args:
        rag_chain (object): The rag chain object
        query (str): The user query
    return:
        response (str): The response from the chatbot
    """
    try:
        st.session_state['chat_history'].append(HumanMessage(content=query))
        response=rag_chain.invoke({
            "input":query,'chat_history':st.session_state['chat_history']
        })

        st.session_state['chat_history'].append(SystemMessage(content=response['answer']))

        return response['answer']
            
    except Exception as e:
        print(str(e))


def load_previous_chat():
    for i in range(len(st.session_state['chat_history'])):
        if i%2==0:
            st.write(f'💬 You: {st.session_state['chat_history'][i].content}') 
        else:
            st.write(f'🤖 Bot: {st.session_state['chat_history'][i].content}')  
    