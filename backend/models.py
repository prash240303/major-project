# File: backend/models.py
from pydantic import BaseModel
from typing import List, Optional
from fastapi import HTTPException


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