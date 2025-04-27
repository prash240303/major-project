# AI Chatbot with FastAPI, Streamlit, and Groq

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- Groq API Key

## Groq Setup
1. Create a Groq account at https://console.groq.com/
2. Obtain your API key from the Groq Console
3. Set the API key as an environment variable:
   ```bash
   # On Unix/macOS
   export GROQ_API_KEY='your_groq_api_key_here'
   
   # On Windows (PowerShell)
   $env:GROQ_API_KEY='your_groq_api_key_here'
   ```

## Installation Steps

### 1. Create a Virtual Environment (Recommended)
```bash
# Create a virtual environment
python3 -m venv chatbot_env

# Activate the virtual environment
# On Windows
chatbot_env\Scripts\activate
# On macOS/Linux
source chatbot_env/bin/activate
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Prepare PDF Document
- Place your PDF document as `doc.pdf` in the project directory

### 4. Running the Application

#### Start FastAPI Backend
```bash
# Ensure GROQ_API_KEY is set
python app.py
# Or 
uvicorn app:app --reload
```

#### Start Streamlit Frontend
```bash
# In a separate terminal
streamlit run streamlit_app.py
```

## Troubleshooting
- Verify your Groq API key is correctly set
- Ensure all dependencies are installed
- Check that the PDF document is present and readable
- Make sure no other services are using ports 8000 (FastAPI) and 8501 (Streamlit)

## Available Groq Models
- `llama3-8b-8192`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

You can change the model in the `app.py` file by modifying the `model` parameter in the `ChatGroq` initialization.




groq_api_key = os.getenv("GROQ_API_KEY", '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')  # Groq API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", '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')  # Set Tavily API key
