from fastapi import APIRouter, HTTPException, BackgroundTasks, Path, Query, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, List
import uuid
import json
import asyncio
from pydantic import BaseModel
from datetime import datetime

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import create_rag_chain, qna
from app.services.chain_cache import get_chain_cache
from app.database import (
    insert_chat_data_async, fetch_chat_history_async, fetch_session_chat_history, 
    clear_history_async, load_or_initialize_json, save_json_to_file, store_chat_history
)
from app.utils.embeddings import get_embeddings
from app.config import JSON_FILE_PATH
from app.auth.jwt import get_current_user, User

router = APIRouter()

class ChatMessage(BaseModel):
    question: str
    
@router.post("/{username}/{session_id}/chat", response_model=ChatResponse)
async def chat(
    request: ChatMessage,
    username: str = Path(..., description="Username for this chat"),
    session_id: str = Path(..., description="Session ID for this chat"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user)
):
    """
    Chat with the AI based on processed documents using specific username and session ID
    
    Requires authentication with JWT token. Users can chat in any session that 
    belongs to their username.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only chat using your own account"
        )
        
    try:
        # Get chat history for this session
        chat_history = await fetch_session_chat_history(username, session_id)
        
        # Get or create cached RAG chain for this user
        chain_cache = get_chain_cache()
        rag_chain, knowledge_base = await chain_cache.get_or_create_chain(username)
        # print(type(rag_chain), type(knowledge_base))
        # try:
        #     print("inside rag chain invoke")
        #     print(rag_chain.invoke({"input": "What is the capital of India", 
        #         "chat_history": [{"role": "user", "content": "hello"}]}))
        # except Exception as e:
        #     print("Exception in invoking rag chain:", str(e))
        if rag_chain is None or knowledge_base is None:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG chain")
        
        # Invoke RAG chain
        try:
            # print("before res")
            
            response = rag_chain.invoke({
                "input": request.question, 
                "chat_history": chat_history[-6:] if chat_history else []
            })


            
            answer = response['answer']
            source_docs = response['documents'] if 'documents' in response else []
        except Exception as e:
            # Fallback to simple QnA if RAG chain fails
            # print("Exception", str(e))
            # print("Exception rag chain fails", str(e))
            answer, _, source_docs = qna(request.question, knowledge_base)
        
        # Store chat in both JSON for reference
        data = load_or_initialize_json(JSON_FILE_PATH)
        
        # Initialize users structure if needed
        if "users" not in data:
            data["users"] = {}
        if username not in data["users"]:
            data["users"][username] = {"sessions": [], "chat_history": {}}
        
        # Ensure session is tracked for this user
        if "sessions" not in data["users"][username]:
            data["users"][username]["sessions"] = []
        if session_id not in data["users"][username]["sessions"]:
            data["users"][username]["sessions"].append(session_id)
        
        # Use the store_chat_history function to save chat data
        store_chat_history(request.question, answer, data, username, session_id)
        save_json_to_file(data, JSON_FILE_PATH)
        
        # Also store in SQLite database asynchronously
        background_tasks.add_task(
            insert_chat_data_async, 
            request.question, 
            answer, 
            username, 
            session_id
        )
        
        return ChatResponse(
            answer=answer,
            source_documents=source_docs,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
@router.get("/{username}/history")
async def get_all_chat_history(
    username: str = Path(..., description="Username to fetch history for"),
    current_user: User = Depends(get_current_user)
):
    """
    Get all chat sessions for a specific user
    
    Requires authentication with JWT token.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only access your own chat history"
        )
    
    try:
        # Get all sessions for user
        return await fetch_chat_history_async(username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@router.get("/{username}/{session_id}/history")
async def get_session_history(
    username: str = Path(..., description="Username to fetch history for"),
    session_id: str = Path(..., description="Session ID to fetch history for"),
    current_user: User = Depends(get_current_user)
):
    """
    Get chat history for a specific session
    
    Requires authentication with JWT token. Users can access any session history 
    that belongs to their username.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only access your own chat history"
        )
    
    try:
        # Get history for specific session
        chat_messages = await fetch_session_chat_history(username, session_id)
        
        # Convert LangChain message objects to dictionaries
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", 
             "content": msg.content}
            for i, msg in enumerate(chat_messages)
        ]
        return {"messages": messages, "session_id": session_id, "username": username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session history: {str(e)}")

@router.delete("/{username}/history")
async def delete_all_history(
    username: str = Path(..., description="Username to clear history for"),
    current_user: User = Depends(get_current_user)
):
    """
    Delete all chat history for a specified user
    
    Requires authentication with JWT token.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only delete your own chat history"
        )
    
    return await clear_history_async(username)

@router.delete("/{username}/{session_id}/history")
async def delete_session_history(
    username: str = Path(..., description="Username to clear history for"),
    session_id: str = Path(..., description="Session ID to clear"),
    current_user: User = Depends(get_current_user)
):
    """
    Delete chat history for a specific session
    
    Requires authentication with JWT token. Users can delete any session history 
    that belongs to their username.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only delete your own chat history"
        )
    
    return await clear_history_async(username, session_id)

@router.post("/{username}/{session_id}/chat/stream")
async def chat_stream(
    request: ChatMessage,
    username: str = Path(..., description="Username for this chat"),
    session_id: str = Path(..., description="Session ID for this chat"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user)
):
    """
    Stream chat responses with the AI based on processed documents using specific username and session ID
    
    Requires authentication with JWT token. Users can chat in any session that 
    belongs to their username. Responses are streamed token by token.
    """
    # Verify authenticated user matches requested username
    if current_user.username != username:
        raise HTTPException(
            status_code=403, 
            detail="You can only chat using your own account"
        )
    
    async def generate_stream():
        try:
            # Get chat history for this session
            chat_history = await fetch_session_chat_history(username, session_id)
            
            # Get or create cached streaming RAG chain for this user
            chain_cache = get_chain_cache()
            streaming_chain, knowledge_base = await chain_cache.get_or_create_streaming_chain(username)
            
            if streaming_chain is None or knowledge_base is None:
                error_data = {
                    "error": "Failed to initialize RAG chain",
                    "finished": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Stream the response
            full_answer = ""
            source_docs = []
            
            try:
                async for chunk in streaming_chain({
                    "input": request.question,
                    "chat_history": chat_history[-6:] if chat_history else []
                }):
                    if chunk.get("finished", False):
                        # Final chunk - store the chat history
                        full_answer = chunk["full_response"]
                        source_docs = chunk.get("documents", [])
                        
                        # Store chat in JSON and database in background
                        background_tasks.add_task(
                            store_chat_complete,
                            request.question,
                            full_answer,
                            username,
                            session_id
                        )
                        
                        # Send final response
                        final_data = {
                            "token": "",
                            "full_response": full_answer,
                            "source_documents": source_docs,
                            "session_id": session_id,
                            "finished": True
                        }
                        yield f"data: {json.dumps(final_data)}\n\n"
                    else:
                        # Stream individual tokens
                        stream_data = {
                            "token": chunk.get("token", ""),
                            "full_response": chunk.get("full_response", ""),
                            "finished": False
                        }
                        yield f"data: {json.dumps(stream_data)}\n\n"
                        
            except Exception as e:
                print("Exception in streaming RAG chain:", str(e))
                # Fallback to simple QnA if streaming fails
                try:
                    from app.services.rag_service import qna
                    answer, _, source_docs = qna(request.question, knowledge_base)
                    
                    # Send fallback response as a single chunk
                    fallback_data = {
                        "token": answer,
                        "full_response": answer,
                        "source_documents": source_docs,
                        "session_id": session_id,
                        "finished": True,
                        "fallback": True
                    }
                    yield f"data: {json.dumps(fallback_data)}\n\n"
                    
                    # Store the fallback response
                    background_tasks.add_task(
                        store_chat_complete,
                        request.question,
                        answer,
                        username,
                        session_id
                    )
                except Exception as fallback_error:
                    error_data = {
                        "error": f"Failed to generate response: {str(fallback_error)}",
                        "finished": True
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
        except Exception as e:
            error_data = {
                "error": f"Error processing request: {str(e)}",
                "finished": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

async def store_chat_complete(question: str, answer: str, username: str, session_id: str):
    """Helper function to store completed chat interaction"""
    try:
        # Store in JSON
        data = load_or_initialize_json(JSON_FILE_PATH)
        
        # Initialize users structure if needed
        if "users" not in data:
            data["users"] = {}
        if username not in data["users"]:
            data["users"][username] = {"sessions": [], "chat_history": {}}
        
        # Ensure session is tracked for this user
        if "sessions" not in data["users"][username]:
            data["users"][username]["sessions"] = []
        if session_id not in data["users"][username]["sessions"]:
            data["users"][username]["sessions"].append(session_id)
        
        # Store chat data
        store_chat_history(question, answer, data, username, session_id)
        save_json_to_file(data, JSON_FILE_PATH)
        
        # Store in database
        await insert_chat_data_async(question, answer, username, session_id)
        
    except Exception as e:
        print(f"Error storing chat data: {str(e)}")
