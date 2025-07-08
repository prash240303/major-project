# File: backend/utils.py
import os
import uuid
import redis
import json
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from config import CONTACT_INFO
from models import Message
from typing import List, Tuple, Optional

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    decode_responses=True
)

# Rate limiting configuration
DAILY_REQUEST_LIMIT = 10
RATE_LIMIT_WINDOW = 24 * 60 * 60  # 24 hours in seconds

def get_client_ip(request):
    """Extract client IP from request headers"""
    # Check for common headers that contain the real IP
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded_for.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host

def get_rate_limit_key(ip_address: str) -> str:
    """Generate Redis key for rate limiting"""
    today = datetime.now().strftime('%Y-%m-%d')
    return f"rate_limit:{ip_address}:{today}"

def check_rate_limit(ip_address: str) -> Tuple[bool, int, Optional[str]]:
    """
    Check if the IP address has exceeded the rate limit
    
    Args:
        ip_address: Client IP address
        
    Returns:
        Tuple of (is_allowed, remaining_requests, error_message)
    """
    try:
        key = get_rate_limit_key(ip_address)
        
        # Get current request count
        current_count = redis_client.get(key)
        
        if current_count is None:
            # First request of the day
            remaining = DAILY_REQUEST_LIMIT - 1
            return True, remaining, None
        
        current_count = int(current_count)
        
        if current_count >= DAILY_REQUEST_LIMIT:
            # Rate limit exceeded
            return False, 0, f"Daily rate limit of {DAILY_REQUEST_LIMIT} requests exceeded. Please try again tomorrow."
        
        # Calculate remaining requests
        remaining = DAILY_REQUEST_LIMIT - current_count - 1
        return True, remaining, None
        
    except Exception as e:
        print(f"[ERROR] Redis rate limit check failed: {e}")
        # If Redis fails, allow the request but log the error
        return True, DAILY_REQUEST_LIMIT - 1, None

def increment_rate_limit(ip_address: str) -> bool:
    """
    Increment the rate limit counter for an IP address
    
    Args:
        ip_address: Client IP address
        
    Returns:
        bool: True if successfully incremented, False otherwise
    """
    try:
        key = get_rate_limit_key(ip_address)
        
        # Use pipeline for atomic operations
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, RATE_LIMIT_WINDOW)
        pipe.execute()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Redis rate limit increment failed: {e}")
        return False

def get_rate_limit_info(ip_address: str) -> dict:
    """
    Get rate limit information for an IP address
    
    Args:
        ip_address: Client IP address
        
    Returns:
        dict: Rate limit information
    """
    try:
        key = get_rate_limit_key(ip_address)
        current_count = redis_client.get(key)
        
        if current_count is None:
            current_count = 0
        else:
            current_count = int(current_count)
        
        remaining = max(0, DAILY_REQUEST_LIMIT - current_count)
        
        # Get TTL for the key
        ttl = redis_client.ttl(key)
        if ttl == -1:  # Key exists but has no expiration
            ttl = RATE_LIMIT_WINDOW
        elif ttl == -2:  # Key doesn't exist
            ttl = RATE_LIMIT_WINDOW
        
        reset_time = datetime.now() + timedelta(seconds=ttl)
        
        return {
            'limit': DAILY_REQUEST_LIMIT,
            'remaining': remaining,
            'used': current_count,
            'reset_time': reset_time.isoformat(),
            'ttl_seconds': ttl
        }
        
    except Exception as e:
        print(f"[ERROR] Redis rate limit info failed: {e}")
        return {
            'limit': DAILY_REQUEST_LIMIT,
            'remaining': DAILY_REQUEST_LIMIT,
            'used': 0,
            'reset_time': (datetime.now() + timedelta(days=1)).isoformat(),
            'ttl_seconds': RATE_LIMIT_WINDOW
        }

def reset_rate_limit(ip_address: str) -> bool:
    """
    Reset rate limit for an IP address (admin function)
    
    Args:
        ip_address: Client IP address
        
    Returns:
        bool: True if successfully reset, False otherwise
    """
    try:
        key = get_rate_limit_key(ip_address)
        redis_client.delete(key)
        return True
        
    except Exception as e:
        print(f"[ERROR] Redis rate limit reset failed: {e}")
        return False

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

def create_fallback_response(message):
    """Create a standardized fallback response with contact information"""
    return f"""{message}
    {CONTACT_INFO}
    """

def get_conversation_history(conversation_id, conversation_store):
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

def generate_uuid():
    """Generate a unique identifier"""
    return str(uuid.uuid4())