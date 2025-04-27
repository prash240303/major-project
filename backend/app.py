import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import shutil
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# Guardrails imports
from guardrails import Guard
from guardrails.hub import (
    SimilarToDocument,
    GibberishText,
    ProvenanceLLM,
    ProvenanceEmbeddings
)


# todo : 
# update the fallback message, add dummy contact and email 
# update the UI 
# create new dashboard for file upload 
# enable excel sheet file fetching 
# fix the text extraction from the files that involve images 

load_dotenv()

app = FastAPI()
# markdarshak 

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend URL
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

# Ensure pdf_files directory exists
PDF_DIR = "pdf_files"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)
    print(f"[INFO] Created directory: {PDF_DIR}")

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

class UploadResponse(BaseModel):
    success: bool
    message: str
    files_processed: int = 0


def load_pdfs_from_directory(directory=PDF_DIR):
    """Load PDFs from local directory using LangChain's PyPDFLoader and split text with RecursiveCharacterTextSplitter"""
    pdf_documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"[WARN] No PDF files found in {directory}")
        return pdf_documents

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(directory, pdf_filename)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()  # Each page is a Document
            split_docs = splitter.split_documents(pages)
            for doc in split_docs:
                doc.metadata["source"] = pdf_filename  # ensure consistent metadata
            pdf_documents.extend(split_docs)
            print(f"[INFO] Loaded and split {pdf_filename} into {len(split_docs)} chunks.")
        except Exception as e:
            print(f"[ERROR] Failed to load {pdf_filename}: {str(e)}")

    print(f"[INFO] Total document chunks created: {len(pdf_documents)}")
    return pdf_documents


def initialize_vector_store():
    """Initialize or reinitialize the vector store from documents"""
    global db, retriever
    
    # Load PDFs from local directory
    documents = load_pdfs_from_directory()
    
    if documents:
        # Initialize Chroma with the documents
        db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
        
        # Configure retriever with a proper configuration
        # Note: Removed the score_threshold parameter which was causing the error
        retriever = db.as_retriever(
            search_kwargs={
                "k": 5  # Return top 5 most relevant documents
            }
        )
        print(f"[INFO] Chroma vector store initialized with {len(documents)} documents.")
        return True
    else:
        print("[ERROR] No documents available to create vector store")
        return False

def create_guardrails(documents):
    """Create and configure the guardrails for response validation"""
    try:
        # Create the embed function for documents
        def embed_function(texts):
            if isinstance(texts, str):
                texts = [texts]
            # Use the same embeddings model as the vector store
            vectors = embeddings.embed_documents(texts)
            import numpy as np
            return np.array(vectors)
        
        # Create a custom LLM callable that uses Groq with Llama3 (Grok)
        def grok_llm_callable(prompt):
            from langchain_groq import ChatGroq
            
            grok_model = ChatGroq(
                model="llama3-8b-8192",  # Using Grok's Llama3 model
                api_key=GROQ_API_KEY,
                temperature=0.1
            )
            
            validation_prompt = f"""
            You are a strict document validator checking if a given response is fully supported by specific context sources.
            Your job is to identify if the response contains ANY information not directly from or implied by the context.
            
            Here is the context source: 
            {prompt}
            
            Reply with ONLY 'yes' if the text is fully supported by the context source, or 'no' if it contains ANY information not directly supported.
            Even small details not in the context should result in a 'no'.
            """
            
            response = grok_model.invoke(validation_prompt).content.strip().lower()
            # Return only "yes" or "no"
            if "yes" in response:
                return "yes"
            else:
                return "no"
        
        # Create guardrails with document validation - stricter settings
        guard = Guard().use_many(
            SimilarToDocument(
                document=documents,
                threshold=0.8,  # Increased threshold
                model="llama3-8b-8192"  # Using Grok's Llama3 model
            ),
            GibberishText(
                threshold=0.6,
                validation_method="sentence",
                on_fail="reject"  # Changed to reject
            ),
            ProvenanceLLM(
                validation_method="sentence",
                llm_callable=grok_llm_callable,  # Using our custom Grok callable
                top_k=3,
                max_tokens=2,
                on_fail="reject"  # Changed to reject
            ),
            ProvenanceEmbeddings(
                threshold=0.85,  # Increased threshold
                validation_method="sentence",
                on_fail="reject"  # Changed to reject
            )
        )
        
        print("[INFO] Enhanced guardrails initialized successfully using Grok (Llama3)")
        return guard
    except Exception as e:
        print(f"[ERROR] Failed to initialize guardrails: {str(e)}")
        return None

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
    """Process a chat request and generate a response with guardrails validation"""
    global retriever, conversation_store
    
    # Ensure vector store is initialized
    if retriever is None:
        success = initialize_vector_store()
        if not success:
            return {"answer": "The knowledge base is not available. Please upload PDF files first.", 
                    "conversation_id": request.conversation_id or str(uuid.uuid4())}

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
        # Retrieve relevant documents
        documents = retriever.invoke(request.question)
        print(f"[INFO] Retrieved {len(documents)} documents for question: {request.question}")

        # Check if the question is relevant to our document context
        is_relevant, rejection_reason = check_question_relevance(request.question, documents)
        
        # If the question is not relevant to our documents, provide a clear rejection message
        if not is_relevant:
            answer = f"I can only answer questions about the documents in my knowledge base. {rejection_reason} Please ask a question related to the content of the uploaded PDFs."
            conversation_history.append(Message(role="assistant", content=answer))
            conversation_store[conversation_id] = conversation_history
            return {"answer": answer, "conversation_id": conversation_id}
            
        # Generate response for relevant content
        context = "\n".join([doc.page_content for doc in documents])
        document_texts = [doc.page_content for doc in documents]
        
        # Include conversation history in the prompt
        conversation_context = format_conversation_history(conversation_history[:-1])  # Exclude current question
        
        prompt = f"""
            You are a document Q&A assistant that ONLY answers questions based on the provided documents.

            **Context:**  
            {context}
            
            {conversation_context}

            **STRICT INSTRUCTIONS:**  
            - You must ONLY answer questions related to the specific document context provided above.
            - If the question isn't directly related to the provided documents, respond ONLY with: 
              "I can only answer questions about the documents in my knowledge base. Your question appears to be unrelated to the documents. Please ask a question related to the content of the uploaded PDFs."
            - Never attempt to answer questions outside of the document context, even if you know the answer.
            - Don't use any knowledge that isn't explicitly contained in the provided documents.
            - Avoid discussing topics not in the provided context, even if they seem related.
            - Provide a detailed, accurate answer based SOLELY on the context.  
            - Avoid copying the text directly; rephrase in a conversational tone.  
            - If you cannot find a direct answer in the context, say: "I cannot find information about that in the provided documents."

            **Question:** {request.question}

            **Response:**"""

        # Generate response using Grok (Llama3)
        response = llm.invoke(prompt).content.strip()
        print(f"[INFO] Generated response with Grok (Llama3): {response[:100]}...")  # Log first 100 characters

        # Enhanced guardrails validation using Grok
        try:
            guard = create_guardrails(document_texts)
            if guard:
                def embed_function(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    vectors = embeddings.embed_documents(texts)
                    import numpy as np
                    return np.array(vectors)

                # Stricter validation settings
                validation_result = guard.validate(
                    response,
                    metadata={
                        "sources": document_texts,
                        "embed_function": embed_function,
                        "pass_on_invalid": False,  # Don't pass if invalid
                        "threshold": 0.75  # Increase threshold for stricter validation
                    }
                )

                if not validation_result.validated:
                    print(f"[WARN] Response failed guardrails validation: {validation_result.failures}")
                    answer = "I can only provide information from the documents in my knowledge base. I don't have enough relevant information to answer your question accurately. Please ask something related to the uploaded documents."
                else:
                    print("[INFO] Response passed guardrails validation")
                    answer = response
            else:
                # If guardrails failed to initialize, apply a basic check
                # Look for phrases that suggest going beyond the document context
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
                    answer = "I can only provide information from the documents in my knowledge base. Please ask a question related to the content of the uploaded PDFs."
                else:
                    answer = response
        except Exception as e:
            print(f"[ERROR] Guardrails validation failed: {str(e)}")
            # Default to a safer response
            answer = "I can only answer questions directly related to the documents in my knowledge base. Please try a more specific question about the document content."

        # Update conversation history with assistant's response
        conversation_history.append(Message(role="assistant", content=answer))
        
        # Store updated conversation history
        conversation_store[conversation_id] = conversation_history
        
        # Log conversation length for debugging
        print(f"[INFO] Conversation {conversation_id} now has {len(conversation_history)} messages")
        
        return {"answer": answer, "conversation_id": conversation_id}
    
    except Exception as e:
        print(f"[ERROR] Error processing chat request: {str(e)}")
        return {
            "answer": "I encountered an error processing your request. Please try again or ask a different question.",
            "conversation_id": conversation_id
        }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(files: List[UploadFile] = File(...)):
    """Upload PDF files to local directory"""
    files_processed = 0
    
    try:
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                # Save file to local directory
                file_path = os.path.join(PDF_DIR, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                files_processed += 1
                print(f"[INFO] Saved {file.filename} to {PDF_DIR}")
        
        # Reinitialize vector store after uploading new files
        success = initialize_vector_store()
        
        return {
            "success": success,
            "message": f"Successfully processed and uploaded {files_processed} files" if success 
                      else "Files uploaded but failed to reinitialize knowledge base",
            "files_processed": files_processed
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on application startup"""
    initialize_vector_store()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)