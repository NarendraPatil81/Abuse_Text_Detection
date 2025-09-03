import os
import logging
from typing import List, Optional
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils.embeddings import get_embeddings, get_vision_llm
from app.utils.image_utils import image_extract, image_summary_doc
from app.config import INDEX_FILE_PATH, INDEX_PATH, IMAGE_OUTPUT_PATH, JSON_FILE_PATH
from app.database import load_or_initialize_json, save_json_to_file, update_processing_status, update_file_processing_timestamp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_documents(
    file_paths: List[str], 
    file_names: List[str], 
    process_images: bool = False,
    session_id: Optional[str] = None,
    username: Optional[str] = None  # Add username parameter
):
    """
    Process uploaded documents with user metadata.
    
    Args:
        file_paths: List of temporary file paths
        file_names: List of original file names
        process_images: Whether to process images in PDFs
        session_id: Session ID for tracking
        username: Username for document ownership
    """
    logger.info(f"Processing {len(file_paths)} documents for user {username}")
    
    # Load current data
    data = load_or_initialize_json(JSON_FILE_PATH)
    
    documents = []
    hf = get_embeddings()
    vision_llm = get_vision_llm()
    
    for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
        try:
            logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_name}")
            
            # Update status to in_progress
            update_processing_status(file_name, "start", data)
            save_json_to_file(data, JSON_FILE_PATH)
            
            # Load PDF
            loader = PyPDFLoader(file_path=file_path)
            doc_pages = loader.load()
            
            # Add username and session_id to document metadata
            for doc in doc_pages:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                # Store username in metadata for ownership tracking
                doc.metadata["username"] = username
                doc.metadata["session_id"] = session_id
                doc.metadata["filename"] = file_name
                doc.metadata["upload_time"] = datetime.now().isoformat()
            
            documents.extend(doc_pages)
            logger.info(f"Loaded {len(doc_pages)} pages from {file_name} with user metadata")
            
            # Process images if requested
            if process_images:
                directory_name = os.path.join(IMAGE_OUTPUT_PATH, file_name.split(".")[0])
                os.makedirs(directory_name, exist_ok=True)
                
                # Extract and process images
                image_extract(file_path, directory_name)
                image_documents = image_summary_doc(directory_name, vision_llm)
                
                # Add metadata to image documents too
                for img_doc in image_documents:
                    if not hasattr(img_doc, 'metadata'):
                        img_doc.metadata = {}
                    img_doc.metadata["username"] = username
                    img_doc.metadata["session_id"] = session_id
                    img_doc.metadata["filename"] = file_name
                    img_doc.metadata["upload_time"] = datetime.now().isoformat()
                    img_doc.metadata["content_type"] = "image"
                    
                documents.extend(image_documents)
                logger.info(f"Processed images from {file_name} and added to documents len{len(image_documents)}")
                
            # Clean up temporary file after processing
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing temp file {file_path}: {str(e)}")
                
            # Mark as completed
            update_processing_status(file_name, "complete", data)
            # Record processing timestamp
            update_file_processing_timestamp(file_name, data)
            save_json_to_file(data, JSON_FILE_PATH)
            logger.info(f"Completed processing {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {str(e)}")
            # Mark as failed
            update_processing_status(file_name, "fail", data)
            save_json_to_file(data, JSON_FILE_PATH)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Make sure metadata is preserved in chunks
    for chunk in chunks:
        if not hasattr(chunk, 'metadata'):
            chunk.metadata = {}
        if 'username' not in chunk.metadata and username:
            chunk.metadata['username'] = username
        if 'session_id' not in chunk.metadata and session_id:
            chunk.metadata['session_id'] = session_id
    
    # Add to vector store
    try:
        if os.path.exists(INDEX_FILE_PATH):
            knowledge_base = FAISS.load_local(
                INDEX_PATH, 
                index_name='AZ', 
                embeddings=hf,
            )
            knowledge_base.add_documents(chunks)
            knowledge_base.save_local(INDEX_PATH, index_name='AZ')
        else:
            knowledge_base = FAISS.from_documents(chunks, hf)
            knowledge_base.save_local(INDEX_PATH, index_name='AZ')
        
        logger.info("Documents processed and added to vector store successfully")
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
    
    logger.info("Document processing completed")
    return True
