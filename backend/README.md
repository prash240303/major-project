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

#nomic api key 

```bash
# Install required packages
nomic login <api_key>
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





# Document Q&A API - Refactored Structure

## Project Structure

The application has been refactored into a modular structure for better maintainability and separation of concerns:

```
project/
├── main.py                 # FastAPI app with routes only
├── config.py              # Configuration and environment setup
├── models.py              # Pydantic models
├── utils.py               # Utility functions
├── document_loader.py     # Document loading from S3
├── vector_store.py        # Vector store management
├── chat_service.py        # Chat logic and response generation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Module Descriptions

### `main.py`
- Contains only the FastAPI application instance
- Defines API endpoints (`/chat`, `/status`, `/refresh`, `/health`)
- Handles CORS configuration
- Minimal business logic - delegates to service modules

### `config.py`
- Environment variable loading
- Configuration constants (CORS origins, directories, contact info)
- Initialization of LLM and embedding models
- Centralized configuration management

### `models.py`
- Pydantic models for request/response schemas
- `Message`, `QuestionRequest`, `ChatResponse`, `SystemStatusResponse`
- Clean separation of data models

### `utils.py`
- Utility functions used across modules
- PDF metadata extraction
- Conversation history management
- Response formatting helpers
- UUID generation

### `document_loader.py`
- S3 document loading logic
- PDF processing and chunking
- Excel file processing for Q&A pairs
- Document conversion to LangChain format

### `vector_store.py`
- Vector store initialization and management
- Chroma database operations
- Retriever configuration
- Global state management for vector store

### `chat_service.py`
- Main chat processing logic
- LLM response generation
- Conversation state management
- Document relevance checking
- Response validation

## Key Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Maintainability**: Changes to one aspect don't affect others
3. **Testability**: Each module can be tested independently
4. **Reusability**: Functions can be easily reused across different parts of the application
5. **Scalability**: Easy to add new features or modify existing ones

## Environment Variables Required

```bash
GROQ_API_KEY=your_groq_api_key
AWS_S3_BUCKET_NAME=your_s3_bucket_name
# Standard AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
```

## Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file

3. Run the application:
```bash
python main.py
```

## API Endpoints

- `POST /chat` - Process chat requests
- `GET /status` - Get system status
- `GET /refresh` - Refresh knowledge base
- `GET /health` - Health check

## Migration Notes

- All functionality remains the same
- API endpoints are unchanged
- Environment variables are the same
- The refactoring only improves code organization
- No breaking changes to existing integrations

## Future Enhancements

With this modular structure, you can easily:
- Add new document types by extending `document_loader.py`
- Implement different LLM providers by modifying `config.py`
- Add new API endpoints in `main.py`
- Implement caching mechanisms in `vector_store.py`
- Add authentication middleware in `main.py`
- Implement logging across all modules