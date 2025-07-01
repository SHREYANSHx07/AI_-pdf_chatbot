# MultiPDF Chat App with Google Gemini

A Python application that allows you to chat with multiple PDF documents using Google's Gemini AI. Ask questions about your PDFs using natural language, and get intelligent responses based on the document content.

## ✨ Features

- 📄 **Multiple PDF Support**: Upload and process multiple PDF documents simultaneously
- 🤖 **Gemini AI Integration**: Powered by Google's advanced Gemini Pro model
- 💬 **Natural Language Chat**: Ask questions in plain English
- 🧠 **Contextual Memory**: Maintains conversation history for better context
- 🎨 **Modern UI**: Clean and intuitive Streamlit interface
- 🔍 **Semantic Search**: Uses advanced embeddings for accurate document retrieval

## 🚀 How It Works

1. **PDF Loading**: Extracts text content from uploaded PDF documents
2. **Text Chunking**: Divides text into manageable chunks for processing
3. **Vector Embeddings**: Creates semantic representations using HuggingFace embeddings
4. **Similarity Matching**: Finds relevant document sections based on your questions
5. **AI Response**: Gemini AI generates intelligent answers using the relevant context

## **Local host** :-
http://localhost:8501/

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🛠️ Installation

1. **Clone the repository**:
   \`\`\`bash
   git clone <repository-url>
   cd multipdf-chat-gemini
   \`\`\`

2. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Set up environment variables**:
   Create a \`.env\` file in the project root and add your Google API key:
   \`\`\`
   GOOGLE_API_KEY=your_google_api_key_here
   \`\`\`

## 🎯 Usage

1. **Start the application**:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

2. **Upload PDFs**:
   - Use the sidebar to upload one or more PDF files
   - Click "Process" to analyze the documents

3. **Start Chatting**:
   - Ask questions about your documents in the main chat interface
   - The AI will provide answers based on the PDF content

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM application framework
- **langchain-google-genai**: Google Gemini integration
- **PyPDF2**: PDF text extraction
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **python-dotenv**: Environment variable management

## 🔧 Configuration

### Environment Variables

- \`GOOGLE_API_KEY\`: Your Google API key for Gemini access

### Model Settings

The application uses:
- **LLM**: Google Gemini Pro
- **Embeddings**: HuggingFace Instructor-XL
- **Vector Store**: FAISS
- **Text Splitter**: Character-based with 1000 chunk size

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Chat Interface**: User and bot message bubbles
- **File Upload**: Drag-and-drop PDF upload
- **Processing Status**: Real-time feedback during document processing
- **Error Handling**: Clear error messages and troubleshooting tips

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure your Google API key is valid
   - Check that the key is properly set in the \`.env\` file

2. **PDF Processing Error**:
   - Verify PDFs contain extractable text (not just images)
   - Try with smaller PDF files first

3. **Memory Issues**:
   - For large PDFs, consider reducing chunk size
   - Process fewer documents at once

### Getting Help

- Check the [Google AI documentation](https://ai.google.dev/)
- Verify your API key has the necessary permissions
- Ensure all dependencies are properly installed

## 🚀 Deployment

### Local Development
\`\`\`bash
streamlit run app.py
\`\`\`

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your \`GOOGLE_API_KEY\` in the secrets management
4. Deploy!

### Docker
\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## 🙏 Acknowledgments

- Google for providing the Gemini AI API
- LangChain for the excellent framework
- Streamlit for the web application framework
- HuggingFace for the embedding models

## 📞 Support

For support or questions:
- Open an issue on GitHub
- Check the documentation links provided
- Review the troubleshooting section

---

**Happy chatting with your PDFs! 🚀**
\`\`\`
