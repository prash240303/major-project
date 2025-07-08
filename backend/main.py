# File: backend/main.py
import uvicorn
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import configuration and models
from config import CORS_ORIGINS, PDF_DIR, EXCEL_DIR, AWS_S3_BUCKET_NAME
from models import QuestionRequest, ChatResponse, SystemStatusResponse

# Import services
from chat_service import process_chat_request
from vector_store import initialize_vector_store, get_retriever, is_vector_store_initialized

# Import rate limiting utilities
from utils import (
    get_client_ip, 
    check_rate_limit, 
    increment_rate_limit, 
    get_rate_limit_info
)

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
async def chat_endpoint(request: QuestionRequest, req: Request):
    """Process a chat request and generate a response"""
    
    # Get client IP
    client_ip = get_client_ip(req)
    print(f"[INFO] Chat request from IP: {client_ip}")
    
    # Check rate limit
    is_allowed, remaining, error_message = check_rate_limit(client_ip)
    
    if not is_allowed:
        # Return rate limit exceeded response
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": error_message,
                "rate_limit_info": get_rate_limit_info(client_ip)
            },
            headers={
                "X-RateLimit-Limit": "10",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(get_rate_limit_info(client_ip)["reset_time"]),
                "Retry-After": str(get_rate_limit_info(client_ip)["ttl_seconds"])
            }
        )
    
    try:
        # Process the chat request
        response = process_chat_request(request)
        
        # Increment rate limit counter only after successful processing
        increment_rate_limit(client_ip)
        
        # Add rate limit headers to response
        rate_info = get_rate_limit_info(client_ip)
        
        # Create response with rate limit headers
        json_response = JSONResponse(
            content=response.dict(),
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(max(0, rate_info["remaining"] - 1)),
                "X-RateLimit-Reset": rate_info["reset_time"]
            }
        )
        
        return json_response
        
    except Exception as e:
        print(f"[ERROR] Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rate-limit-info")
async def get_user_rate_limit_info(req: Request):
    """Get rate limit information for the current user"""
    client_ip = get_client_ip(req)
    rate_info = get_rate_limit_info(client_ip)
    
    return {
        "ip": client_ip,
        "rate_limit": rate_info
    }

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