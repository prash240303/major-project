import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import boto3
from typing import List
from botocore.exceptions import ClientError
from io import BytesIO
from PyPDF2 import PdfReader

import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable")

# Initialize embedding model
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="remote")

# Initialize LLM - using Grok (Llama3) model exclusively
llm = ChatGroq(
    model="llama3-8b-8192",  # Using Grok's Llama3 model 
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# Vector store
db = None
retriever = None

# Conversation memory store
conversation_store = {}

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


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    messages: Optional[List[Message]] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    source_link_metadata: Optional[str] = None  

class SystemStatusResponse(BaseModel):
    status: str
    pdf_count: int
    excel_count: int
    knowledge_base_initialized: bool

# s3 bucket Directory paths
PDF_DIR = "pdf_files"
EXCEL_DIR = "excel_files"

def extract_pdf_metadata(pdf_path):
    """
    Extract metadata from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        dict: Dictionary containing PDF metadata
    """
    try:
        print(f"\n[CONSOLE] Starting metadata extraction for {pdf_path}")
        metadata = {}
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            # Log raw metadata object type
            print(f"[CONSOLE] PDF metadata type: {type(pdf_reader.metadata)}")
            print(f"[CONSOLE] PDF metadata raw: {pdf_reader.metadata}")
            
            # Direct check for Description
            if '/Description' in pdf_reader.metadata:
                description_value = pdf_reader.metadata['/Description']
                print(f"[CONSOLE] Found /Description directly: {description_value}")
                
            # Extract metadata from the PDF
            if pdf_reader.metadata:
                # Convert PyPDF2 metadata to a regular Python dictionary
                for key in pdf_reader.metadata:
                    # Log each key-value pair
                    print(f"[CONSOLE] Processing metadata key: {key}, value: {pdf_reader.metadata[key]}")
                    
                    # Clean up the key format (remove leading '/' if present)
                    clean_key = key.replace('/', '') if isinstance(key, str) and key.startswith('/') else key
                    metadata[clean_key] = pdf_reader.metadata[key]
                    
                    # Also store the original key format
                    metadata[key] = pdf_reader.metadata[key]
                
                # Check if we have a Description field now
                if 'Description' in metadata:
                    print(f"[CONSOLE] After processing, found Subject: {metadata['Subject']}")
                else:
                    print(f"[CONSOLE] After processing, no Subject field found in metadata")
                    
                # Debug: Print all extracted metadata
                print(f"[DEBUG] Extracted PDF metadata: {metadata}")
                
        return metadata
    
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF metadata: {str(e)}")
        return {}


def load_pdfs_from_s3() -> List[Document]:
    """
    Load the pdf files from s3, where in the bucket, "/pdf-files" folder the PDFs are stored,
    fetch those files using boto3 and convert them to LangChain Document objects
    that can be used for the vector store db.
    
    Returns:
        List[Document]: List of Document objects created from pdf files.
    """
    try:
        # Initialize boto3 S3 client
        s3 = boto3.client('s3')
        
        # Get the bucket name from environment variable
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not bucket_name:
            print("[ERROR] AWS_S3_BUCKET_NAME environment variable not set")
            return []
        
        # List objects in the PDF_DIR prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{PDF_DIR}/")
        
        if 'Contents' not in response:
            print(f"[INFO] No PDF files found in s3://{bucket_name}/{PDF_DIR}/")
            return []
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Process each PDF file
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory objects or non-PDF files
            if not key.lower().endswith('.pdf'):
                continue
                
            print(f"\n[CONSOLE] Processing PDF from S3: {key}")
            
            try:
                # Create a temporary file
                temp_file_path = f"/tmp/{uuid.uuid4()}.pdf"
                
                # Download the file from S3
                s3.download_file(bucket_name, key, temp_file_path)
                print(f"[CONSOLE] Downloaded PDF to temporary path: {temp_file_path}")
                
                # Extract PDF metadata using PyPDF2 before loading with PyPDFLoader
                print(f"[CONSOLE] Extracting metadata from {temp_file_path}")
                pdf_metadata = extract_pdf_metadata(temp_file_path)
                
                # Debug: Print the extracted metadata 
                print(f"[CONSOLE] Complete PDF metadata for {key}: {pdf_metadata}")
                
                # Check specifically for the Description field
                source_link = None
                # Check all possible Description key variations
                description_keys = ['Description', '/Description', 'description']
                for desc_key in description_keys:
                    if desc_key in pdf_metadata and pdf_metadata[desc_key]:
                        source_link = pdf_metadata[desc_key]
                        print(f"[CONSOLE] Found source link using key '{desc_key}': {source_link}")
                        break
                
                if not source_link:
                    print(f"[CONSOLE] No Description/source link found in metadata")
                
                # Load the PDF using PyPDFLoader
                print(f"[CONSOLE] Loading PDF content with PyPDFLoader")
                loader = PyPDFLoader(temp_file_path)
                pdf_docs = loader.load()
                print(f"[CONSOLE] Loaded {len(pdf_docs)} pages from PDF")
                
                # Split the documents
                split_docs = text_splitter.split_documents(pdf_docs)
                print(f"[CONSOLE] Split into {len(split_docs)} document chunks")
                
                # Add metadata to identify the source
                for i, doc in enumerate(split_docs):
                    doc.metadata["source"] = key
                    doc.metadata["document_type"] = "pdf"
                    
                    # Add the source link from the PDF metadata if available
                    if source_link:
                        # Store the link in multiple fields to ensure it's found later
                        doc.metadata["source_link"] = source_link
                        doc.metadata["Description"] = source_link
                        doc.metadata["/Description"] = source_link
                        doc.metadata["description"] = source_link
                        print(f"[CONSOLE] Added source link to chunk {i} metadata")
                    
                    # Add all metadata fields for completeness
                    for meta_key, meta_value in pdf_metadata.items():
                        doc.metadata[meta_key] = str(meta_value)
                    
                    # Debug the first chunk's metadata
                    if i == 0:
                        print(f"[CONSOLE] Sample document chunk metadata: {doc.metadata}")
                
                documents.extend(split_docs)
                
                # Clean up temp file
                os.remove(temp_file_path)
                print(f"[CONSOLE] Removed temporary file {temp_file_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process PDF {key}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        print(f"[INFO] Successfully loaded {len(documents)} document chunks from {len(response['Contents'])} PDF files in S3")
        return documents
        
    except ClientError as e:
        print(f"[ERROR] AWS S3 client error: {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in load_pdfs_from_s3: {str(e)}")
        return []


def load_pdfs_from_s3() -> List[Document]:
    """
    Load the pdf files from s3, where in the bucket, "/pdf-files" folder the PDFs are stored,
    fetch those files using boto3 and convert them to LangChain Document objects
    that can be used for the vector store db.
    
    Returns:
        List[Document]: List of Document objects created from pdf files.
    """
    try:
        # Initialize boto3 S3 client
        s3 = boto3.client('s3')
        
        # Get the bucket name from environment variable
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not bucket_name:
            print("[ERROR] AWS_S3_BUCKET_NAME environment variable not set")
            return []
        
        # List objects in the PDF_DIR prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{PDF_DIR}/")
        
        if 'Contents' not in response:
            print(f"[INFO] No PDF files found in s3://{bucket_name}/{PDF_DIR}/")
            return []
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Process each PDF file
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory objects or non-PDF files
            if not key.lower().endswith('.pdf'):
                continue
                
            print(f"[INFO] Processing PDF from S3: {key}")
            
            try:
                # Create a temporary file
                temp_file_path = f"/tmp/{uuid.uuid4()}.pdf"
                
                # Download the file from S3
                s3.download_file(bucket_name, key, temp_file_path)
                
                # Extract PDF metadata using PyPDF2 before loading with PyPDFLoader
                pdf_metadata = extract_pdf_metadata(temp_file_path)
                source_link = None
                
                # Check if we have a description field that contains a URL
                if "Description" in pdf_metadata:
                    source_link = pdf_metadata["Description"]
                elif "/Description" in pdf_metadata:
                    source_link = pdf_metadata["/Description"]
                
                # Load the PDF using PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                pdf_docs = loader.load()
                
                # Split the documents
                split_docs = text_splitter.split_documents(pdf_docs)
                
                # Add metadata to identify the source
                for doc in split_docs:
                    doc.metadata["source"] = key
                    doc.metadata["document_type"] = "pdf"
                    
                    # Add the source link from the PDF metadata if available
                    if source_link:
                        doc.metadata["source_link"] = source_link  # Use a consistent key "source_link"
                    
                    # Add other metadata fields if needed
                    for meta_key, meta_value in pdf_metadata.items():
                        # Convert PyPDF2 metadata keys (which often have '/' prefix) to clean keys
                        clean_key = meta_key.replace('/', '') if meta_key.startswith('/') else meta_key
                        doc.metadata[clean_key.lower()] = str(meta_value)
                
                documents.extend(split_docs)
                
                # Clean up temp file
                os.remove(temp_file_path)
                
            except Exception as e:
                print(f"[ERROR] Failed to process PDF {key}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        print(f"[INFO] Successfully loaded {len(documents)} document chunks from {len(response['Contents'])} PDF files in S3")
        return documents
        
    except ClientError as e:
        print(f"[ERROR] AWS S3 client error: {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in load_pdfs_from_s3: {str(e)}")
        return []

def load_excel_froms3() -> List[Document]:
    """
    Load question-answer pairs from Excel files stored in S3 (stored in the '/excel_files' directory
    of s3 bucket) directly in memory using boto3 and convert them to LangChain Document objects
    for vector store usage.
    
    Returns:
        List[Document]: List of Document objects created from Excel sheet QA pairs.
    """
    try:
        # Initialize boto3 S3 client
        s3 = boto3.client('s3')
        
        # Get the bucket name from environment variable
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not bucket_name:
            print("[ERROR] AWS_S3_BUCKET_NAME environment variable not set")
            return []
        
        # List objects in the EXCEL_DIR prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{EXCEL_DIR}/")
        
        if 'Contents' not in response:
            print(f"[INFO] No Excel files found in s3://{bucket_name}/{EXCEL_DIR}/")
            return []
        
        documents = []
        
        # Process each Excel file
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory objects or non-Excel files
            if not (key.lower().endswith('.xlsx') or key.lower().endswith('.xls')):
                continue
                
            print(f"[INFO] Processing Excel file from S3: {key}")
            
            try:
                # Get the file from S3 into memory
                response = s3.get_object(Bucket=bucket_name, Key=key)
                excel_data = BytesIO(response['Body'].read())
                
                # Read Excel file with pandas
                df = pd.read_excel(excel_data)
                
                # Ensure required columns exist
                required_columns = ['question', 'answer']
                if not all(col in df.columns for col in required_columns):
                    print(f"[WARN] Excel file {key} missing required columns (question, answer). Skipping.")
                    continue
                
                # Process each row in the Excel file
                for idx, row in df.iterrows():
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()
                    
                    # Skip rows with empty questions or answers
                    if not question or not answer or question.lower() == 'nan' or answer.lower() == 'nan':
                        continue
                    
                    # Create a document from this QA pair
                    # Use the question as the page_content to match against user queries
                    doc = Document(
                        page_content=question,
                        metadata={
                            "source": key,
                            "document_type": "excel",
                            "question": question,
                            "answer": answer,
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
            
            except Exception as e:
                print(f"[ERROR] Failed to process Excel file {key}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        print(f"[INFO] Successfully loaded {len(documents)} QA pairs from Excel files in S3")
        return documents
        
    except ClientError as e:
        print(f"[ERROR] AWS S3 client error: {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in load_excel_froms3: {str(e)}")
        return []

def initialize_vector_store():
    """Initialize or reinitialize the vector store from both PDF documents and Excel QA pairs"""
    global db, retriever
    
    # Load PDFs from local directory
    pdf_documents = load_pdfs_from_s3()
    
    # Load Excel QA data as documents
    excel_documents = load_excel_froms3()
    # Combine all documents
    all_documents = pdf_documents + excel_documents
    
    if all_documents:
        # Initialize Chroma with the combined documents
        db = Chroma.from_documents(all_documents, embeddings, persist_directory="./chroma_db")
        
        # Configure retriever with a proper configuration
        retriever = db.as_retriever(
            search_kwargs={
                "k": 5  # Return top 5 most relevant documents
            }
        )
        print(f"[INFO] Chroma vector store initialized with {len(all_documents)} documents "
              f"({len(pdf_documents)} PDF chunks and {len(excel_documents)} Excel QA pairs).")
        return True
    else:
        print("[WARN] No PDF documents or Excel QA pairs available to create vector store")
        return False


def create_fallback_response(message):
    """Create a standardized fallback response with contact information"""
    return f"""{message}
    --------------------------------------------------
For further contact regarding admission queries:

**Dr. Vickram Jeet Singh**
Associate Dean Academic (Undergraduate Programmes)
**Email**: as.daug@nitj.ac.in
**Phone**: 0181-5037542
**Languages**: English, Hindi, Punjabi
--------------------------------------------------
"""


def get_conversation_history(conversation_id):
    """Retrieve conversation history by ID"""
    if conversation_id and conversation_id in conversation_store:
        return conversation_store[conversation_id]
    return []


def format_conversation_history(messages):
    """Format conversation history for the LLM prompt"""
    if not messages:
        return ""
    
    formatted = "\n\n**Previous Conversation:**\n"
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    return formatted


def check_question_relevance(question, documents):
    """Check if the question is relevant to the document context"""
    # If no documents were retrieved, it's likely irrelevant
    if not documents or len(documents) == 0:
        return False, "No relevant documents found for this question."
    
    # Check document relevance - the logic here depends on what metadata is available
    # For simplicity, if we find at least 2 documents, consider it relevant
    if len(documents) >= 2:
        return True, ""
    
    # Default to allowing the question if we're unsure
    return True, ""


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QuestionRequest):
    """Process a chat request and generate a response"""
    global retriever, conversation_store
    
    # Ensure vector store is initialized
    if retriever is None:
        success = initialize_vector_store()
        if not success:
            fallback_message = create_fallback_response("The knowledge base is not available. Please ensure there are PDF or Excel files in the designated directories.")
            return {"answer": fallback_message, "conversation_id": request.conversation_id or str(uuid.uuid4()), "source_links": [], "source_link_metadata": None}

    # Get or create conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Get conversation history
    conversation_history = get_conversation_history(conversation_id)
    
    # Update with new messages if provided
    if request.messages:
        conversation_history = request.messages
    
    # Add current user message
    conversation_history.append(Message(role="user", content=request.question))
    
    try:
        # Retrieve relevant documents from the unified vector store
        documents = retriever.invoke(request.question)
        print(f"[INFO] Retrieved {len(documents)} documents for question: {request.question}")

        # Check if the question is relevant to our document context
        is_relevant, rejection_reason = check_question_relevance(request.question, documents)
        
        # If the question is not relevant to our documents, provide a clear rejection message with contact info
        if not is_relevant:
            answer = create_fallback_response(f"I can only answer questions about the documents in my knowledge. Please contact the officials below for further queries.")
            conversation_history.append(Message(role="assistant", content=answer))
            conversation_store[conversation_id] = conversation_history
            return {"answer": answer, "conversation_id": conversation_id, "source_links": [], "source_link_metadata": None}
        
        # Extract source links and subject metadata from the retrieved documents
        source_link_metadata = None
        
        # Print all document metadata for debugging
        print(f"\n[CONSOLE] Document metadata analysis for question: {request.question}")
        for i, doc in enumerate(documents):
            print(f"[CONSOLE] Document {i} metadata full dump: {doc.metadata}")
        
        # Extract subject metadata - take the first valid one we find
        print(f"\n[CONSOLE] Looking for subject metadata")
        for i, doc in enumerate(documents):
            # Check for Subject fields with various casing and formats
            subject_keys = ["Subject", "/Subject", "subject"]
            for key in subject_keys:
                if key in doc.metadata and doc.metadata[key]:
                    source_link_metadata = doc.metadata[key]
                    print(f"[CONSOLE] Document {i}: Found subject '{source_link_metadata}' in key '{key}'")
                    break
            
            # If we found a subject, stop looking
            if source_link_metadata:
                break
                
 
        print(f"[CONSOLE] Final extracted subject metadata: {source_link_metadata}")
            
        # Check if we have Excel QA documents in the retrieved documents
        excel_qa_documents = [doc for doc in documents if doc.metadata.get("document_type") == "excel"]
        
        # If we have Excel QA documents and they're highly relevant (first in results), use those directly
        if excel_qa_documents and documents[0].metadata.get("document_type") == "excel":
            # Extract the answer from the most relevant Excel QA document
            top_qa_doc = excel_qa_documents[0]
            excel_answer = top_qa_doc.metadata.get("answer")
            
            if excel_answer:
                print(f"[INFO] Using direct answer from Excel QA document for question: {request.question}")
                answer = excel_answer
                
                # Update conversation history with assistant's response
                conversation_history.append(Message(role="assistant", content=answer))
                conversation_store[conversation_id] = conversation_history
                
                # Return answer with source links and subject metadata if available
                return {
                    "answer": answer, 
                    "conversation_id": conversation_id,
                    "source_link_metadata": source_link_metadata
                }
        
        # Generate response for relevant content using all retrieved documents
        context = "\n".join([doc.page_content for doc in documents])
        
        # Include conversation history in the prompt
        conversation_context = format_conversation_history(conversation_history[:-1])  # Exclude current question
        
        # Update the prompt template
        prompt = f"""
            You are a professional document Q&A assistant that provides precise responses exclusively based on the document context provided.

            **Document Context:**  
            {context}
            
            {conversation_context}

            **CRITICAL INSTRUCTIONS:**  
            - Only provide information that is explicitly contained in the document context above
            - If the question cannot be answered using only the provided context, respond with: 
            "I don't have information about that in my knowledge. Please contact the officials below for further queries.
            For further contact regarding admission queries:
            Dr. Vickram Jeet Singh
            Associate Dean Academic (Undergraduate Programmes)
            Email: as.daug@nitj.ac.in
            Phone: 0181-5037542
            Languages: English, Hindi, Punjabi
            "
            - Do not reference yourself as an AI or assistant
            - Do not mention "document context" or "provided documents" in your response
            - Maintain a formal, professional tone
            - Be concise and direct in your answers
            - Never fabricate information or make assumptions beyond what is stated in the context
            - If only partial information is available, clearly state the limitations of what you can provide
            - For any document labeled as Excel QA content, prioritize using the exact answer provided

            **Question:** {request.question}

            **Response:**"""
        
        # Generate response using Grok (Llama3)
        response = llm.invoke(prompt).content.strip()
        print(f"[INFO] Generated response with Grok (Llama3): {response[:100]}...")  # Log first 100 characters

        # Apply a basic check for suspicious phrases that might indicate hallucination
        suspicious_phrases = [
            "based on my knowledge",
            "generally speaking",
            "in general",
            "it is widely known",
            "typically",
            "as an AI",
            "I don't have access",
            "I'm not able to"
        ]
        
        suspicious = False
        for phrase in suspicious_phrases:
            if phrase.lower() in response.lower():
                suspicious = True
                break
        
        if suspicious:
            answer = create_fallback_response("I can only provide information from the documents in my knowledge.")
        else:
            answer = response

        # Update conversation history with assistant's response
        conversation_history.append(Message(role="assistant", content=answer))
        
        # Store updated conversation history
        conversation_store[conversation_id] = conversation_history
        
        # Log source links and subject being returned
        print(f"[INFO] Conversation {conversation_id} now has {len(conversation_history)} messages")
        print(f"[CONSOLE] Final subject metadata being returned: {source_link_metadata}")
        
        # Return answer with source links and subject metadata if available
        return {
            "answer": answer, 
            "conversation_id": conversation_id,
            "source_link_metadata": source_link_metadata
        }
    
    except Exception as e:
        print(f"[ERROR] Error processing chat request: {str(e)}")
        return {
            "answer": create_fallback_response("I encountered an error processing your request. Please try again or ask a different question."),
            "conversation_id": conversation_id,
            "source_links": [],
            "source_link_metadata": None
        }
    
@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status including file count information"""
    try:
        # Count PDF files
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        pdf_count = len(pdf_files)
        
        # Count Excel files
        excel_files = [f for f in os.listdir(EXCEL_DIR) if f.lower().endswith(('.xlsx', '.xls'))]
        excel_count = len(excel_files)
        
        # Check if knowledge base is initialized
        kb_initialized = retriever is not None
        
        return {
            "status": "ok",
            "pdf_count": pdf_count,
            "excel_count": excel_count,
            "knowledge_base_initialized": kb_initialized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/refresh", response_model=SystemStatusResponse)
async def refresh_knowledge_base():
    """Force refresh of the knowledge base by reloading all files"""
    try:
        success = initialize_vector_store()
        
        # Count PDF files
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        pdf_count = len(pdf_files)
        
        # Count Excel files
        excel_files = [f for f in os.listdir(EXCEL_DIR) if f.lower().endswith(('.xlsx', '.xls'))]
        excel_count = len(excel_files)
        
        return {
            "status": "refreshed" if success else "failed",
            "pdf_count": pdf_count,
            "excel_count": excel_count,
            "knowledge_base_initialized": retriever is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on application startup"""
    initialize_vector_store()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)