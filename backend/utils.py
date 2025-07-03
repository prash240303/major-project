import os
import uuid
from PyPDF2 import PdfReader
from config import CONTACT_INFO
from models import Message
from typing import List


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