import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os


from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR("hkunlp/instructor-large")
# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create vector store from text chunks using HuggingFace embeddings"""
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversation chain using Gemini LLM"""
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("AIzaSyAftIATi4htA6xjvTEV74bRhu3fWvYbVhA"),
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Handle user input and display conversation"""
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        st.error("Please make sure your Google API key is valid and you have processed the PDFs.")

def main():
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="Chat with multiple PDFs", 
        page_icon=":books:",
        layout="wide"
    )
    
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main header
    st.header("Chat with multiple PDFs :books:")
    st.markdown("*Powered by Google Gemini AI*")
    
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    # User input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process PDFs first!")

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Extract text from PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded PDFs.")
                            return
                        
                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.info(f"Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        st.success("Documents processed successfully! You can now ask questions.")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Display information about uploaded files
        if pdf_docs:
            st.subheader("Uploaded Files:")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")

if __name__ == '__main__':
    main()
