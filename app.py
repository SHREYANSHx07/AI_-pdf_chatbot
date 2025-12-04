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

# ---------- Dependency imports with clear error messages ----------
try:
    from PyPDF2 import PdfReader
except ImportError as e:
    st.error(f"Failed to import PyPDF2: {e}")
    st.error("Please check the requirements.txt file and ensure PyPDF2 is installed.")
    st.stop()

# LangChain and related packages (robust import block)
try:
    # new splitters package
    from langchain_text_splitters import CharacterTextSplitter
except ImportError as e:
    st.error(f"Failed to import CharacterTextSplitter: {e}")
    st.error("Install with: pip install langchain-text-splitters")
    st.stop()

try:
    # core langchain memory & chains
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ImportError as e:
    st.error(f"Failed to import langchain core modules: {e}")
    st.error("Install with: pip install langchain")
    st.stop()

try:
    # community vectorstores & embeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"Failed to import langchain_community modules: {e}")
    st.error("Install with: pip install langchain-community")
    st.stop()

try:
    # Google GenAI wrapper (third-party)
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Failed to import ChatGoogleGenerativeAI: {e}")
    st.error("Install the package you use to access Google Generative AI (pip install <package-name>)")
    st.stop()

# html templates for UI (make sure this file exists)
try:
    from htmlTemplates import css, bot_template, user_template
except ImportError as e:
    st.error(f"Failed to import htmlTemplates: {e}")
    st.error("Make sure htmlTemplates.py is in the same directory as app.py")
    st.stop()

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # not critical
    pass


# ---------- Helper functions ----------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks for processing"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    # CharacterTextSplitter provides split_text or split_documents depending on version.
    # We'll try split_text first, fallback to split_documents if available.
    try:
        return text_splitter.split_text(text)
    except Exception:
        try:
            # some versions expect a list of docs
            return text_splitter.split_documents([{"page_content": text}])
        except Exception:
            # last resort: naive chunking
            chunks = []
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunks.append(text[start:end])
                start = end - chunk_overlap if end - chunk_overlap > start else end
            return chunks


def get_vectorstore(text_chunks, hf_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create vector store from text chunks using HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)
    # Some versions accept `embedding` or `embeddings` argument name, so we try both.
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except TypeError:
        vectorstore = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create conversation chain using Gemini LLM (Google Generative)"""
    # Try to get API key from Streamlit secrets first (for deployment), then from environment
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("⚠️ Please set your GOOGLE_API_KEY in Streamlit secrets or environment variables")
        st.stop()

    # Create LLM wrapper
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # adjust model name as desired/available
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Build conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def _display_chat_history(chat_history):
    """
    Display chat history robustly.
    chat_history might be a list of message objects (with .content),
    or list of dicts/tuples, or plain strings.
    We'll handle common cases.
    """
    if not chat_history:
        return

    for i, message in enumerate(chat_history):
        # determine content
        content = None
        # message could be a langchain Message object with .content
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict) and ("content" in message):
            content = message["content"]
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            # some chat histories are (role, content)
            content = str(message[1])
        else:
            content = str(message)

        # Alternate user/bot template based on index (even user, odd bot) — adapt if your templates assume otherwise
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


def handle_userinput(user_question):
    """Handle user input and display conversation"""
    try:
        # ConversationalRetrievalChain expects a dict with "question" key
        response = st.session_state.conversation({"question": user_question})
        # response may contain 'answer' and 'chat_history' or 'result' etc.
        chat_history = response.get("chat_history") or response.get("chat_history") or st.session_state.get("chat_history")
        # store chat history if available
        if chat_history is not None:
            st.session_state.chat_history = chat_history
        # show the final answer (if present)
        final_answer = response.get("answer") or response.get("result") or response.get("response") or response.get("output")
        if final_answer:
            # print bot answer using bot template (append at end)
            st.write(bot_template.replace("{{MSG}}", str(final_answer)), unsafe_allow_html=True)

        # display the full history (if available)
        _display_chat_history(st.session_state.get("chat_history"))
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        st.error("Please make sure your Google API key is valid and you have processed the PDFs.")


# ---------- Streamlit app ----------
def main():
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
        layout="wide"
    )

    # render CSS (htmlTemplates)
    try:
        st.write(css, unsafe_allow_html=True)
    except Exception:
        # if css missing or invalid, ignore styling
        pass

    # Session state initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    st.markdown("*Powered by Google Gemini AI*")

    # Check if Google API key is available (from Streamlit secrets or environment)
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("⚠️ Please set your GOOGLE_API_KEY")
        st.info("For local development: Add to .env file")
        st.info("For Streamlit Cloud: Add to secrets management")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        # Don't return immediately — allow file upload so user can process docs locally without LLM
        # return

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process PDFs first!")

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
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded PDFs.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            st.info(f"Created {len(text_chunks)} text chunks")

                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)

                            st.success("Documents processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")

        if pdf_docs:
            st.subheader("Uploaded Files:")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")


if __name__ == '__main__':
    main()

