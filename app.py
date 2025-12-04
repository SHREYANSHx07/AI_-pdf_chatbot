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

# ---------- Dependency imports with robust / backward-compatible fallbacks ----------

# PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError as e:
    st.error(f"Failed to import PyPDF2: {e}")
    st.error("Please ensure PyPDF2 is installed (pip install PyPDF2).")
    st.stop()

# CharacterTextSplitter: prefer dedicated package, fall back to older langchain location
try:
    from langchain_text_splitters import CharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter
    except Exception as e:
        st.error(f"Failed to import CharacterTextSplitter: {e}")
        st.error("Install with: pip install langchain-text-splitters")
        st.stop()

# ConversationBufferMemory: try new split package first, then classic langchain
try:
    # preferred for new split architecture (if installed)
    from langchain_core.memory import ConversationBufferMemory
except Exception:
    try:
        # common/older layout
        from langchain.memory import ConversationBufferMemory
    except Exception as e:
        st.error(f"Failed to import ConversationBufferMemory from langchain_core or langchain: {e}")
        st.error("Ensure you have a compatible langchain or langchain-core installed.")
        st.stop()

# ConversationalRetrievalChain: try langchain.chains
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception as e:
    st.error(f"Failed to import ConversationalRetrievalChain: {e}")
    st.error("Install a compatible langchain package (pip install langchain).")
    st.stop()

# FAISS vectorstore & HuggingFaceEmbeddings: try community package first, fallback to langchain top-level
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception as e:
        st.error(f"Failed to import FAISS or HuggingFaceEmbeddings: {e}")
        st.error("Install langchain-community or a compatible langchain (pip install langchain-community).")
        st.stop()

# Google GenAI wrapper (third-party) — adjust if your package name differs
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:
    st.error(f"Failed to import ChatGoogleGenerativeAI: {e}")
    st.error("Install the package you use to access Google Generative AI (e.g. pip install langchain-google-genai) or change import to your wrapper.")
    st.stop()

# html templates for UI (make sure this file exists)
try:
    from htmlTemplates import css, bot_template, user_template
except Exception as e:
    st.error(f"Failed to import htmlTemplates: {e}")
    st.error("Make sure htmlTemplates.py is in the same directory as app.py and exports css, bot_template, user_template.")
    st.stop()

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------- Helper functions ----------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None
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
    # Try multiple splitter APIs
    try:
        return text_splitter.split_text(text)
    except Exception:
        try:
            docs = [{"page_content": text}]
            split = text_splitter.split_documents(docs)
            chunks = []
            for d in split:
                if isinstance(d, dict) and "page_content" in d:
                    chunks.append(d["page_content"])
                elif hasattr(d, "page_content"):
                    chunks.append(d.page_content)
                else:
                    chunks.append(str(d))
            return chunks
        except Exception:
            # fallback naive chunking
            chunks = []
            start = 0
            text_len = len(text)
            if text_len == 0:
                return []
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunks.append(text[start:end])
                if end == text_len:
                    break
                start = end - chunk_overlap
            return chunks


def get_vectorstore(text_chunks, hf_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create vector store from text chunks using HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)

    # Try different FAISS constructors depending on installed version
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except TypeError:
        try:
            vectorstore = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
        except Exception:
            try:
                embs = embeddings.embed_documents(text_chunks)
                # from_embeddings may require different arg order; we try common forms
                try:
                    vectorstore = FAISS.from_embeddings(embs, text_chunks)
                except TypeError:
                    vectorstore = FAISS.from_embeddings(embs, text_chunks, metadatas=None)
            except Exception as e:
                st.error(f"Failed to create FAISS vectorstore: {e}")
                st.stop()
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create conversation chain using Gemini LLM (Google Generative)"""
    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("⚠️ Please set your GOOGLE_API_KEY in Streamlit secrets or environment variables")
        st.stop()

    # Build LLM wrapper (try common constructor signatures)
    llm = None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
    except TypeError:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=api_key,
                temperature=0.3
            )
        except Exception as e:
            st.error(f"Failed to construct ChatGoogleGenerativeAI: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to construct ChatGoogleGenerativeAI: {e}")
        st.stop()

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Build conversational retrieval chain (robust to signature differences)
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    except TypeError:
        try:
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm, vectorstore.as_retriever(), memory=memory
            )
        except Exception as e:
            st.error(f"Failed to build ConversationalRetrievalChain: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to build ConversationalRetrievalChain: {e}")
        st.stop()

    return conversation_chain


def _display_chat_history(chat_history):
    if not chat_history:
        return
    for i, message in enumerate(chat_history):
        content = None
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict) and ("content" in message):
            content = message["content"]
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            content = str(message[1])
        else:
            content = str(message)
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
    except Exception as e:
        st.error(f"Conversation call failed: {e}")
        return

    final_answer = None
    chat_history = None

    if isinstance(response, dict):
        final_answer = response.get("answer") or response.get("result") or response.get("response") or response.get("output")
        chat_history = response.get("chat_history") or response.get("history") or st.session_state.get("chat_history")
    else:
        final_answer = str(response)

    if final_answer:
        st.write(bot_template.replace("{{MSG}}", str(final_answer)), unsafe_allow_html=True)

    if chat_history is not None:
        st.session_state.chat_history = chat_history
        _display_chat_history(chat_history)
    else:
        _display_chat_history(st.session_state.get("chat_history"))


# ---------- Streamlit app ----------
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")

    try:
        st.write(css, unsafe_allow_html=True)
    except Exception:
        pass

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    st.markdown("*Powered by Google Gemini AI*")

    api_key = None
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.warning("⚠️ GOOGLE_API_KEY not found. Add it to secrets or environment variables if you want LLM functionality.")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process PDFs first!")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])

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



