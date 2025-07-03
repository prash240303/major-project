import os
import uuid
import boto3
from typing import Dict, List
from models import Message, QuestionRequest, ChatResponse
from utils import (
    create_fallback_response, 
    get_conversation_history, 
    format_conversation_history, 
    check_question_relevance,
    extract_pdf_metadata,
    generate_uuid
)
from vector_store import get_retriever, initialize_vector_store, is_vector_store_initialized
from config import llm, AWS_S3_BUCKET_NAME

# Conversation memory store
conversation_store: Dict[str, List[Message]] = {}


def process_chat_request(request: QuestionRequest) -> ChatResponse:
    """Process a chat request and generate a response"""
    global conversation_store
    
    # Ensure vector store is initialized
    retriever = get_retriever()
    if not is_vector_store_initialized():
        success = initialize_vector_store()
        if not success:
            fallback_message = create_fallback_response(
                "The knowledge base is not available. Please ensure there are PDF or Excel files in the designated directories."
            )
            return ChatResponse(
                answer=fallback_message, 
                conversation_id=request.conversation_id or generate_uuid(), 
                source_link_metadata=None
            )
        retriever = get_retriever()

    # Get or create conversation ID
    conversation_id = request.conversation_id or generate_uuid()
    
    # Get conversation history
    conversation_history = get_conversation_history(conversation_id, conversation_store)
    
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
            answer = create_fallback_response(
                "I can only answer questions about the documents in my knowledge. Please contact the officials below for further queries."
            )
            conversation_history.append(Message(role="assistant", content=answer))
            conversation_store[conversation_id] = conversation_history
            return ChatResponse(answer=answer, conversation_id=conversation_id, source_link_metadata=None)
        
        # Extract metadata only from PDF documents used as context
        pdf_metadata = _extract_pdf_metadata_from_documents(documents)
        
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
                
                # Return answer with metadata if available
                return ChatResponse(
                    answer=answer, 
                    conversation_id=conversation_id,
                    source_link_metadata=None  # Excel docs don't have PDF metadata
                )
        
        # Generate response for relevant content using all retrieved documents
        answer = _generate_llm_response(request.question, documents, conversation_history)
        
        # Update conversation history with assistant's response
        conversation_history.append(Message(role="assistant", content=answer))
        
        # Store updated conversation history
        conversation_store[conversation_id] = conversation_history
        
        # Get source_link_metadata from pdf_metadata if available
        source_link_metadata = _extract_source_link_from_metadata(pdf_metadata)
        
        # Return answer with metadata if available
        return ChatResponse(
            answer=answer, 
            conversation_id=conversation_id,
            source_link_metadata=source_link_metadata
        )
    
    except Exception as e:
        print(f"[ERROR] Error processing chat request: {str(e)}")
        return ChatResponse(
            answer=create_fallback_response(
                "I encountered an error processing your request. Please try again or ask a different question."
            ),
            conversation_id=conversation_id,
            source_link_metadata=None
        )


def _extract_pdf_metadata_from_documents(documents):
    """Extract PDF metadata from retrieved documents"""
    pdf_metadata = None
    s3 = boto3.client('s3')
    bucket_name = AWS_S3_BUCKET_NAME
    
    # Find PDF documents in the retrieved documents
    pdf_docs = [doc for doc in documents if doc.metadata.get("document_type") == "pdf"]
    
    if pdf_docs:
        # Use the first PDF document's source to get metadata
        pdf_source = pdf_docs[0].metadata.get("source")
        
        if pdf_source and bucket_name:
            try:
                # Create a temporary file to store the PDF
                temp_file_path = f"/tmp/{uuid.uuid4()}.pdf"
                
                # Download the file from S3
                s3.download_file(bucket_name, pdf_source, temp_file_path)
                print(f"[CONSOLE] Downloaded PDF to temporary path: {temp_file_path}")
                
                # Extract PDF metadata
                pdf_metadata = extract_pdf_metadata(temp_file_path)
                print(f"[CONSOLE] Extracted metadata for context PDF: {pdf_metadata}")
                
                # Clean up temp file
                os.remove(temp_file_path)
                
            except Exception as e:
                print(f"[ERROR] Failed to extract PDF metadata: {str(e)}")
    
    return pdf_metadata


def _generate_llm_response(question: str, documents, conversation_history: List[Message]) -> str:
    """Generate response using LLM with document context"""
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
        - Do not make assumptions or provide general knowledge outside the document context
        - Do not include any information that is not directly supported by the document context
        - If there is any greetings or pleasantries in the question, respond with a simple acknowledgment like "Hello" or "Hi" without any additional information
        - Do not provide any personal opinions or interpretations
        - Do not include any disclaimers or statements about your capabilities
        - Do not reference yourself as an AI or assistant
        - Do not mention "document context" or "provided documents" in your response
        - Maintain a formal, professional tone
        - Be concise and direct in your answers
        - Never fabricate information or make assumptions beyond what is stated in the context
        - If only partial information is available, clearly state the limitations of what you can provide
        - For any document labeled as Excel QA content, prioritize using the exact answer provided

        **Question:** {question}

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
        return create_fallback_response("I can only provide information from the documents in my knowledge.")
    else:
        return response


def _extract_source_link_from_metadata(pdf_metadata):
    """Extract source link from PDF metadata"""
    source_link_metadata = None
    if pdf_metadata:
        description_keys = ['Description', '/Description', 'description']
        for key in description_keys:
            if key in pdf_metadata and pdf_metadata[key]:
                source_link_metadata = pdf_metadata[key]
                print(f"[CONSOLE] Using metadata from source PDF, key '{key}': {source_link_metadata}")
                break
        
        # If no Description found, try Subject
        if not source_link_metadata:
            subject_keys = ['Subject', '/Subject', 'subject']
            for key in subject_keys:
                if key in pdf_metadata and pdf_metadata[key]:
                    source_link_metadata = pdf_metadata[key]
                    print(f"[CONSOLE] Using subject metadata from source PDF, key '{key}': {source_link_metadata}")
                    break
    
    return source_link_metadata