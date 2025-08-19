# 🚀 Deploy to Streamlit Cloud

## Quick Deployment Steps

### 1. Commit and Push Your Code
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

### 3. Connect Your GitHub Repository
- Click "New app" 
- Connect to GitHub and select your repository: `AI_-pdf_chatbot`
- Choose branch: `main`
- Main file path: `app.py`

### 4. Add Your Google API Key
In the Advanced settings:
1. Click on "Secrets"
2. Add this content:
```toml
GOOGLE_API_KEY = "AIzaSyCZKFy3geJwVeZJXGQUrQI9cxYm56p_by4"
```

### 5. Deploy!
Click "Deploy" and wait for your app to build.

## 📋 Requirements Already Set Up
✅ `requirements.txt` - Dependencies list  
✅ `.streamlit/config.toml` - Streamlit configuration  
✅ `app.py` - Updated to work with Streamlit secrets  
✅ GitHub repository connected  

## 🔧 Alternative: Use requirements-deploy.txt
If you encounter dependency issues, you can:
1. Rename `requirements-deploy.txt` to `requirements.txt`
2. Commit and push the changes
3. Redeploy the app

## 📖 After Deployment
Your app will be available at: `https://yourappname.streamlit.app`

## 🆘 Troubleshooting
- **Build fails**: Check the build logs for specific error messages
- **API key issues**: Ensure the secret is properly set in Streamlit Cloud
- **Dependency conflicts**: Try using `requirements-deploy.txt` instead
