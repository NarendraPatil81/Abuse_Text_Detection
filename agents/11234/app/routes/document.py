from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Depends, BackgroundTasks
from typing import List
import os
import shutil
from tempfile import NamedTemporaryFile
from datetime import datetime

from app.models.schemas import UploadResponse
from app.services.document_service import process_documents
from app.database import load_or_initialize_json, save_json_to_file, update_file_count, update_processing_status
from app.config import JSON_FILE_PATH
from app.auth.jwt import get_current_user, User

router = APIRouter()

@router.post("/{username}/upload", response_model=UploadResponse)
async def upload_documents(
    username: str = Path(..., description="Username uploading documents"),
    files: List[UploadFile] = File(...),
    process_images: bool = False,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()  # Add this
):
    """
    Upload and process documents for a specific user.
    
    Requires authentication with JWT token.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only upload documents to your own account"
        )
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a response object to track progress
    # This file stores information about all uploaded files
    uploaded_files_data = load_or_initialize_json(JSON_FILE_PATH)
    processed_files = []
    temp_paths = []
    
    # Initialize users structure if needed
    if "users" not in uploaded_files_data:
        uploaded_files_data["users"] = {}
    if username not in uploaded_files_data["users"]:
        uploaded_files_data["users"][username] = {"files": [], "processed_files": {}}
    
    # Process each file
    for pdf_file in files:
        try:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                shutil.copyfileobj(pdf_file.file, temp_file)
                temp_path = temp_file.name
                temp_paths.append(temp_path)
            
            # Update file count and track upload time
            file_name = pdf_file.filename
            update_file_count(file_name, uploaded_files_data)
            update_processing_status(file_name, "start", uploaded_files_data)
            
            # Add file to user's list with timestamp
            if file_name not in uploaded_files_data["users"][username]["files"]:
                uploaded_files_data["users"][username]["files"].append(file_name)
                
            # Track upload timestamp in user's processed_files
            if "processed_files" not in uploaded_files_data["users"][username]:
                uploaded_files_data["users"][username]["processed_files"] = {}
            
            uploaded_files_data["users"][username]["processed_files"][file_name] = {
                "upload_time": datetime.now().isoformat(),
                "status": "processing"
            }
                
            processed_files.append(file_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process file {pdf_file.filename}: {str(e)}")
        finally:
            pdf_file.file.close()
    
    # Save the updated file tracking data
    save_json_to_file(uploaded_files_data, JSON_FILE_PATH)
    
    # Add to background tasks instead of processing immediately
    background_tasks.add_task(
        process_documents,
        temp_paths, 
        [f.filename for f in files], 
        process_images,
        username=username
    )
    
    return {
        "message": "Files uploaded and queued for processing.",
        "files_processed": len(files),
        "username": username,
        "files_details": {file: uploaded_files_data["files"].get(file, 0) for file in processed_files},
        "status": "processing"  # Indicate files are being processed
    }

@router.get("/{username}/documents")
async def list_user_documents(
    username: str = Path(..., description="Username to list documents for"),
    current_user: User = Depends(get_current_user)
):
    """
    List all documents uploaded by a specific user.
    
    Requires authentication with JWT token.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only view documents from your own account"
        )
    
    data = load_or_initialize_json(JSON_FILE_PATH)
    
    if "users" not in data or username not in data.get("users", {}):
        return {
            "username": username,
            "message": "No documents found for this user",
            "total_files": 0,
            "documents": [],
            "updated_at": datetime.now().isoformat(),
        }
    
    user_data = data["users"][username]
    user_files = user_data.get("files", [])
    
    # Create document list with timestamps
    documents = []
    for file in user_files:
        doc_info = {
            "filename": file,
            "upload_count": data["files"].get(file, 1)
        }
        
        # Add processing timestamps if available
        if "processed_files" in user_data and file in user_data["processed_files"]:
            file_data = user_data["processed_files"][file]
            if isinstance(file_data, dict):
                doc_info.update(file_data)
            else:
                doc_info["processed_date"] = file_data
        
        # Check global processing status
        if "processing_status" in data:
            if file in data["processing_status"]["in_progress"]:
                doc_info["status"] = "processing"
            elif file in data["processing_status"]["completed"]:
                doc_info["status"] = "completed"
            elif file in data["processing_status"]["failed"]:
                doc_info["status"] = "failed"
        
        documents.append(doc_info)
    
    return {
        "username": username,
        "total_files": len(user_files),
        "documents": documents,
        "updated_at": datetime.now().isoformat(),
        # Shows where the data is stored
    }

