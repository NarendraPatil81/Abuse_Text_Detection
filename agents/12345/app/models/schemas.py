from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Chat models
class ChatRequest(BaseModel):
    question: str
    username: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]] = []
    session_id: str

class HistoryRequest(BaseModel):
    username: str

class HistoryItem(BaseModel):
    question: str
    answer: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

# Document processing models
class UploadResponse(BaseModel):
    message: str
    files_processed: int
    files_details: Dict[str, int]

class ProcessingStatusResponse(BaseModel):
    status: str
    in_progress: List[str]
    completed: List[str]
    failed: List[str]
    total_files: int

# Auth models
class LoginRequest(BaseModel):
    username: str
