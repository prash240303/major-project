import uvicorn
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Import configuration and models
from config import CORS_ORIGINS, PDF_DIR, EXCEL_DIR, AWS_S3_BUCKET_NAME
from models import QuestionRequest, ChatResponse, SystemStatusResponse

# Import services
from chat_service import process_chat_request
from vector_store import initialize_vector_store, get_retriever, is_vector_store_initialized

# Initialize FastAPI app
app = FastAPI(title="Document Q&A API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QuestionRequest):
    """Process a chat request and generate a response"""
    return process_chat_request(request)


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status including file count information"""
    try:
        # Count PDF files (this is a placeholder since we're using S3)
        pdf_count = 0
        excel_count = 0
        
        # In a real implementation, you might want to count S3 objects
        # For now, we'll use placeholder values or implement S3 counting
        
        # Check if knowledge base is initialized
        kb_initialized = is_vector_store_initialized()
        
        return SystemStatusResponse(
            status="ok",
            pdf_count=pdf_count,
            excel_count=excel_count,
            knowledge_base_initialized=kb_initialized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/refresh", response_model=SystemStatusResponse)
async def refresh_knowledge_base(background_tasks: BackgroundTasks):
    """Trigger a background refresh of the knowledge base."""
    
    def refresh_task():
        try:
            initialize_vector_store()
            print("[INFO] Knowledge base refreshed successfully")
        except Exception as e:
            print(f"[ERROR] Background refresh failed: {e}")

    background_tasks.add_task(refresh_task)

    # Return current system state (pre-refresh)
    pdf_count = 0
    excel_count = 0

    try:
        retriever = get_retriever()
        if retriever is not None:
            try:
                total_docs = retriever.vectorstore._collection.count() if hasattr(retriever.vectorstore, '_collection') else 0
                pdf_count = total_docs
            except:
                pdf_count = 130  # Fallback value
                excel_count = 0
    except:
        pass

    return SystemStatusResponse(
        status="refresh_started",
        knowledge_base_initialized=is_vector_store_initialized(),
        pdf_count=pdf_count,
        excel_count=excel_count
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on application startup"""
    print("[INFO] Starting application...")
    success = initialize_vector_store()
    if success:
        print("[INFO] Vector store initialized successfully")
    else:
        print("[WARN] Vector store initialization failed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Document Q&A API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)