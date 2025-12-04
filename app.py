# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os

# # Load environment variables (optional for deployment)
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     # dotenv not available in production deployment
#     pass

# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF documents"""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     """Split text into manageable chunks for processing"""
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     """Create vector store from text chunks using HuggingFace embeddings"""
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore):
#     """Create conversation chain using Gemini LLM"""
#     # Initialize Gemini LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         google_api_key=os.getenv("AIzaSyAftIATi4htA6xjvTEV74bRhu3fWvYbVhA"),
#         temperature=0.3,
#         convert_system_message_to_human=True
#     )
    
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', 
#         return_messages=True
#     )
    
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def handle_userinput(user_question):
#     """Handle user input and display conversation"""
#     try:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chat_history = response['chat_history']

#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(user_template.replace(
#                     "{{MSG}}", message.content), unsafe_allow_html=True)
#             else:
#                 st.write(bot_template.replace(
#                     "{{MSG}}", message.content), unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error processing your question: {str(e)}")
#         st.error("Please make sure your Google API key is valid and you have processed the PDFs.")

# def main():
#     # Page configuration
#     st.set_page_config(
#         page_title="Chat with multiple PDFs", 
#         page_icon=":books:",
#         layout="wide"
#     )
    
#     st.write(css, unsafe_allow_html=True)

#     # Initialize session state
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     # Main header
#     st.header("Chat with multiple PDFs :books:")
#     st.markdown("*Powered by Google Gemini AI*")
    
#     # Check if Google API key is set
#     if not os.getenv("GOOGLE_API_KEY"):
#         st.error("⚠️ Please set your GOOGLE_API_KEY in the .env file")
#         st.info("Get your API key from: https://makersuite.google.com/app/apikey")
#         return

#     # User input
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         if st.session_state.conversation is not None:
#             handle_userinput(user_question)
#         else:
#             st.warning("Please upload and process PDFs first!")

#     # Sidebar for PDF upload
#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", 
#             accept_multiple_files=True,
#             type=['pdf']
#         )
        
#         if st.button("Process"):
#             if pdf_docs:
#                 with st.spinner("Processing your documents..."):
#                     try:
#                         # Extract text from PDFs
#                         raw_text = get_pdf_text(pdf_docs)
                        
#                         if not raw_text.strip():
#                             st.error("No text could be extracted from the uploaded PDFs.")
#                             return
                        
#                         # Split text into chunks
#                         text_chunks = get_text_chunks(raw_text)
#                         st.info(f"Created {len(text_chunks)} text chunks")
                        
#                         # Create vector store
#                         vectorstore = get_vectorstore(text_chunks)
                        
#                         # Create conversation chain
#                         st.session_state.conversation = get_conversation_chain(vectorstore)
                        
#                         st.success("Documents processed successfully! You can now ask questions.")
                        
#                     except Exception as e:
#                         st.error(f"Error processing documents: {str(e)}")
#             else:
#                 st.warning("Please upload at least one PDF file.")
        
#         # Display information about uploaded files
#         if pdf_docs:
#             st.subheader("Uploaded Files:")
#             for pdf in pdf_docs:
#                 st.write(f"📄 {pdf.name}")

# if __name__ == '__main__':
#     main()

# import streamlit as st
# import os

# # Import with error handling
# try:
#     from PyPDF2 import PdfReader
# except ImportError as e:
#     st.error(f"Failed to import PyPDF2: {e}")
#     st.error("Please check the requirements.txt file and ensure PyPDF2 is installed.")
#     st.stop()

# try:
#     # from langchain.text_splitter import CharacterTextSplitter
#     from langchain_text_splitters import CharacterTextSplitter
#     from langchain_core.memory import ConversationBufferMemory
#     # from langchain.memory import ConversationBufferMemory
#     from langchain.chains import ConversationalRetrievalChain
#     from langchain_community.vectorstores import FAISS
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     from langchain_google_genai import ChatGoogleGenerativeAI
# except ImportError as e:
#     st.error(f"Failed to import LangChain components: {e}")
#     st.error("Please check the requirements.txt file and ensure all LangChain packages are installed.")
#     st.stop()

# try:
#     from htmlTemplates import css, bot_template, user_template
# except ImportError as e:
#     st.error(f"Failed to import htmlTemplates: {e}")
#     st.error("Make sure htmlTemplates.py is in the same directory as app.py")
#     st.stop()

# # Load environment variables (optional for deployment)
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass


# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF documents"""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     """Split text into manageable chunks for processing"""
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)


# def get_vectorstore(text_chunks):
#     """Create vector store from text chunks using HuggingFace embeddings"""
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     """Create conversation chain using Gemini LLM"""
#     # Try to get API key from Streamlit secrets first (for deployment), then from environment
#     api_key = None
#     try:
#         api_key = st.secrets["GOOGLE_API_KEY"]
#     except (KeyError, FileNotFoundError):
#         api_key = os.getenv("GOOGLE_API_KEY")
    
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=api_key,
#         temperature=0.3,
#         convert_system_message_to_human=True
#     )

#     memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         return_messages=True
#     )

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     """Handle user input and display conversation"""
#     try:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chat_history = response['chat_history']

#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(user_template.replace(
#                     "{{MSG}}", message.content), unsafe_allow_html=True)
#             else:
#                 st.write(bot_template.replace(
#                     "{{MSG}}", message.content), unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error processing your question: {str(e)}")
#         st.error("Please make sure your Google API key is valid and you have processed the PDFs.")


# def main():
#     st.set_page_config(
#         page_title="Chat with multiple PDFs",
#         page_icon=":books:",
#         layout="wide"
#     )

#     st.write(css, unsafe_allow_html=True)

#     # Session state initialization
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     st.markdown("*Powered by Google Gemini AI*")

#     # Check if Google API key is available (from Streamlit secrets or environment)
#     api_key = None
#     try:
#         api_key = st.secrets["GOOGLE_API_KEY"]
#     except (KeyError, FileNotFoundError):
#         api_key = os.getenv("GOOGLE_API_KEY")
    
#     if not api_key:
#         st.error("⚠️ Please set your GOOGLE_API_KEY")
#         st.info("For local development: Add to .env file")
#         st.info("For Streamlit Cloud: Add to secrets management")
#         st.info("Get your API key from: https://makersuite.google.com/app/apikey")
#         return

#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         if st.session_state.conversation is not None:
#             handle_userinput(user_question)
#         else:
#             st.warning("Please upload and process PDFs first!")

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'",
#             accept_multiple_files=True,
#             type=['pdf']
#         )
        
#         if st.button("Process"):
#             if pdf_docs:
#                 with st.spinner("Processing your documents..."):
#                     try:
#                         raw_text = get_pdf_text(pdf_docs)

#                         if not raw_text.strip():
#                             st.error("No text could be extracted from the uploaded PDFs.")
#                             return

#                         text_chunks = get_text_chunks(raw_text)
#                         st.info(f"Created {len(text_chunks)} text chunks")

#                         vectorstore = get_vectorstore(text_chunks)
#                         st.session_state.conversation = get_conversation_chain(vectorstore)

#                         st.success("Documents processed successfully! You can now ask questions.")

#                     except Exception as e:
#                         st.error(f"Error processing documents: {str(e)}")
#             else:
#                 st.warning("Please upload at least one PDF file.")

#         if pdf_docs:
#             st.subheader("Uploaded Files:")
#             for pdf in pdf_docs:
#                 st.write(f"📄 {pdf.name}")


# if __name__ == '__main__':
#     main()
import streamlit as st
import os
import importlib

# ---- Quick import-check helper (shows clear errors in Streamlit UI) ----
def check_import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        st.error(f"Missing dependency or wrong package: {name} — {e}")
        return None

# ---- Essential libs ----
try:
    from PyPDF2 import PdfReader
except Exception as e:
    st.error(f"PyPDF2 import failed: {e}")
    st.stop()

# Prefer the dedicated splitter package; fall back to older langchain path
splitter_mod = None
try:
    from langchain_text_splitters import CharacterTextSplitter
    splitter_mod = "langchain_text_splitters"
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter
        splitter_mod = "langchain.text_splitter"
    except Exception as e:
        st.error(f"CharacterTextSplitter import failed: {e}")
        st.error("Install langchain-text-splitters or a compatible langchain.")
        st.stop()

# Classic langchain memory (this is the required import)
try:
    from langchain.memory import ConversationBufferMemory
except Exception as e:
    st.error(f"Failed to import ConversationBufferMemory: {e}")
    st.error("Ensure the 'langchain' package is installed (and 'langchain-core' is NOT present).")
    st.stop()

# Chains
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception as e:
    st.error(f"Failed to import ConversationalRetrievalChain: {e}")
    st.stop()

# Vectorstore & embeddings (langchain-community)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    st.error(f"Failed to import langchain_community modules: {e}")
    st.error("Install langchain-community (pip install langchain-community).")
    st.stop()

# Google Generative wrapper — ensure the package name you installed matches this import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:
    st.error(f"Failed to import ChatGoogleGenerativeAI: {e}")
    st.error("Install the wrapper you use for Google Generative AI (e.g. langchain-google-genai).")
    st.stop()

# Optional htmlTemplates (if missing, UI will still run but be plain)
try:
    from htmlTemplates import css, bot_template, user_template
except Exception:
    css = ""
    bot_template = "<div class='bot'>{{MSG}}</div>"
    user_template = "<div class='user'>{{MSG}}</div>"

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Helper functions ----
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            try:
                p = page.extract_text()
            except Exception:
                p = None
            if p:
                text += p + "\n"
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    try:
        return splitter.split_text(text)
    except Exception:
        # fallback naive chunking
        chunks = []
        i = 0
        n = len(text)
        if n == 0:
            return []
        while i < n:
            j = min(i + chunk_size, n)
            chunks.append(text[i:j])
            if j == n:
                break
            i = j - chunk_overlap
        return chunks

def get_vectorstore(text_chunks, hf_model="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except TypeError:
        return FAISS.from_texts(texts=text_chunks, embeddings=embeddings)

def get_conversation_chain(vectorstore):
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Set GOOGLE_API_KEY in Streamlit secrets or environment variables.")
        st.stop()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3, convert_system_message_to_human=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def _display_chat_history(chat_history):
    if not chat_history:
        return
    for i, m in enumerate(chat_history):
        content = getattr(m, "content", m if isinstance(m, str) else str(m))
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", content), unsafe_allow_html=True)

def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
    except Exception as e:
        st.error(f"Conversation call failed: {e}")
        return
    answer = None
    history = None
    if isinstance(response, dict):
        answer = response.get("answer") or response.get("result") or response.get("output")
        history = response.get("chat_history") or response.get("history")
    else:
        answer = str(response)
    if answer:
        st.write(bot_template.replace("{{MSG}}", str(answer)), unsafe_allow_html=True)
    if history:
        st.session_state.chat_history = history
        _display_chat_history(history)
    else:
        _display_chat_history(st.session_state.get("chat_history"))

# ---- Streamlit UI ----
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    try:
        st.write(css, unsafe_allow_html=True)
    except Exception:
        pass

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs")
    st.markdown("*Powered by Google Generative AI*")

    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.warning("Set GOOGLE_API_KEY in Streamlit secrets or environment variables to use LLM features.")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process PDFs first.")

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs and click Process", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text extracted from PDFs.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        st.info(f"Created {len(chunks)} text chunks")
                        vs = get_vectorstore(chunks)
                        st.session_state.conversation = get_conversation_chain(vs)
                        st.success("Documents processed. Ask a question!")

        if pdf_docs:
            st.subheader("Uploaded files")
            for p in pdf_docs:
                st.write(f"📄 {p.name}")

if __name__ == "__main__":
    main()



