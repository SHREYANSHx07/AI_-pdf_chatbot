# PDF Reader App - Deployment Guide

## 📋 Requirements Files

### `requirements.txt` (Development - Exact Versions)
- Use for local development
- Contains exact package versions that are working
- Fastest installation for development

### `requirements-deploy.txt` (Production - Flexible Versions)
- Use for deployment to cloud platforms
- Contains version ranges for better compatibility
- Handles platform-specific differences automatically

## 🚀 Deployment Instructions

### Local Development
```bash
# Option 1: Use exact versions (recommended for development)
pip install -r requirements.txt

# Option 2: Use flexible versions
pip install -r requirements-deploy.txt
```

### Cloud Deployment (Heroku, Railway, Render, etc.)
```bash
# Use the deployment-friendly requirements
pip install -r requirements-deploy.txt
```

### Environment Setup
1. Create `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

2. Get your API key from: https://makersuite.google.com/app/apikey

### Running the Application
```bash
streamlit run app.py
```

## 🔧 Platform-Specific Notes

### macOS ARM64 (M1/M2)
- Some packages may show compatibility warnings (grpcio)
- These warnings don't affect functionality
- App will work correctly despite warnings

### Linux/Windows
- All packages should install without warnings
- Use requirements-deploy.txt for maximum compatibility

### Python Version Support
- **Recommended**: Python 3.9 - 3.11
- **Minimum**: Python 3.8
- **Maximum**: Python 3.12

## 📦 Key Dependencies

| Component | Purpose |
|-----------|---------|
| `streamlit` | Web framework |
| `langchain` | AI/ML framework |
| `google-generativeai` | Gemini AI integration |
| `PyPDF2` | PDF processing |
| `faiss-cpu` | Vector search |
| `sentence-transformers` | Text embeddings |

## 🐛 Troubleshooting

### Deployment Errors
1. **grpcio platform warnings**: Ignore - functionality not affected
2. **Version conflicts**: Use `requirements-deploy.txt`
3. **Memory issues**: Use smaller PDF files for testing

### API Issues
1. **Model not found**: Ensure you're using `gemini-1.5-flash` or `gemini-1.5-pro`
2. **API key errors**: Check your `.env` file and API key validity
3. **Rate limits**: Wait a few seconds between requests

## ✅ Verification Steps

After installation:
```bash
# Check for conflicts (warnings are OK)
pip check

# Test imports
python -c "import streamlit; import langchain_google_genai; print('✅ Installation successful')"

# Run the app
streamlit run app.py
```

## 🌐 Supported Deployment Platforms

✅ **Tested Platforms:**
- Heroku
- Railway
- Render
- Streamlit Cloud
- Local development (macOS, Linux, Windows)

✅ **Deployment Tips:**
- Use `requirements-deploy.txt` for cloud platforms
- Set `GOOGLE_API_KEY` in platform environment variables
- Ensure sufficient memory (minimum 1GB recommended)
