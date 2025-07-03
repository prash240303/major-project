import os
from dotenv import load_dotenv
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable")

# Directory paths
PDF_DIR = "pdf_files"
EXCEL_DIR = "excel_files"

# CORS origins
CORS_ORIGINS = [
    "http://localhost:5173", 
    "https://major-project-mqf2.vercel.app",
    "https://dashboard-margdarshak.vercel.app/"
]

# Contact information
CONTACT_INFO = """
--------------------------------------------------
For further contact regarding admission queries:

**Dr. Vickram Jeet Singh**
Associate Dean Academic (Undergraduate Programmes)
**Email**: as.daug@nitj.ac.in
**Phone**: 0181-5037542
**Languages**: English, Hindi, Punjabi
--------------------------------------------------
"""

# Initialize embedding model
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="remote")

# Initialize LLM - using Grok (Llama3) model exclusively
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.2
)