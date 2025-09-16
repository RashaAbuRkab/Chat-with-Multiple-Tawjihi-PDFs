import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css,bot_template, user_template
from langchain.llms import HuggingFaceHub




def get_pdf_text(pdf_docs):
    text ="" 
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-x1") 
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore

def get_conversation_chain(vector_store):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_massages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store,
        memory=memory
        )
    return conversation_chain

def handel_userinput(user_question):
    response = st.session_state.conversation({'question' : user_question})
    st.session_state.chat_history = response['chat_history']

    for i,massege in enumerate(st.session_state.chat_history):
        if i %2 ==0 :
            st.write(user_template.replace("{{MSG}}",massege.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",massege.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs",page_icon = ':books:')
    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state.chat_history:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs  :books:")
    user_question = st.text_input("Ask a Question about your documents: ")
    if user_question:
        handel_userinput(user_question)
    st.write(user_template.replace("{{MSG}}","hello robot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","hello human"),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your Documents: ')
        pdf_docs= st.file_uploader(
            "Upload Your Pdfs here and click on 'Process' "
            ,accept_multiple_files=True)
        
        if st.button('Process'): 
            with st.spinner("Processing:"):   
                # get pdf text
                raw_text=get_pdf_text(pdf_docs)

                # get the text chunks 
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
        st.session_state.conversation    


if __name__ == '__main__':
    main()