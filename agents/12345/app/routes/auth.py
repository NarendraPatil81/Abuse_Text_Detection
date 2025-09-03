from fastapi import APIRouter, HTTPException, status, Body, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from app.auth.jwt import (
    user_exists, 
    create_access_token, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    User,
    get_current_user,
    create_session_id,
    UsernameRequest
)
from app.services.chain_cache import get_chain_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router with explicit tags
router = APIRouter(
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)

@router.post(
    "/login", 
    response_model=Token, 
    summary="Login with username only", 
    description="Provide username to get JWT token",
    response_description="Returns JWT token for authorization"
)
async def login_for_access_token(
    user_request: UsernameRequest,
    background_tasks: BackgroundTasks
):
    """
    Simple login endpoint that accepts username and returns a JWT token.
    No password required for this simple implementation.
    """
    username = user_request.username
    logger.info(f"Login attempt for username: {username}")
    
    # Check if user exists
    if not user_exists(username):
        logger.warning(f"Login failed: user {username} not found")
        raise HTTPException(
            status_code=401,
            detail="Invalid username",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create session ID
    session_id = create_session_id()
    logger.debug(f"Created session ID: {session_id}")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.utcnow() + access_token_expires
    
    try:
        access_token = create_access_token(
            data={"sub": username, "session_id": session_id},
            expires_delta=access_token_expires        )
        logger.info(f"Login successful for {username}, generated token and session ID {session_id[:8]}")
        
        # Pre-warm the RAG chain cache for this user in the background
        chain_cache = get_chain_cache()
        background_tasks.add_task(chain_cache.pre_warm_cache, username)
        logger.info(f"Started pre-warming RAG chain cache for user {username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": username,
            "session_id": session_id,
            "expires_at": expires_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Token creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create access token")

@router.get(
    "/me", 
    response_model=Dict,
    summary="Get current user info",
    description="Get information about the current authenticated user"
)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    ## Get Current User Information
    
    Returns information about the currently authenticated user.
    
    Requires a valid JWT token in the Authorization header as `Bearer {token}`.
    """
    return {
        "username": current_user.username,
        "session_id": current_user.session_id,
        "authenticated": True
    }

@router.get(
    "/cache/stats",
    response_model=Dict,
    summary="Get RAG chain cache statistics",
    description="Get statistics about the RAG chain cache"
)
async def get_cache_stats(current_user: User = Depends(get_current_user)):
    """
    ## Get Cache Statistics
    
    Returns statistics about the RAG chain cache including:
    - Total cached entries
    - Cache size limits
    - Per-user cache information
    
    Requires a valid JWT token in the Authorization header.
    """
    chain_cache = get_chain_cache()
    return chain_cache.get_cache_stats()

@router.post(
    "/cache/warm/{target_username}",
    response_model=Dict,
    summary="Pre-warm cache for a user",
    description="Pre-warm the RAG chain cache for a specific user"
)
async def warm_cache_for_user(
    target_username: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    ## Pre-warm Cache
    
    Pre-warm the RAG chain cache for a specific user.
    Users can only warm their own cache unless they have admin privileges.
    
    Requires a valid JWT token in the Authorization header.
    """
    # Allow users to warm their own cache, or admins to warm any cache
    if current_user.username != target_username and current_user.username != "admin":
        raise HTTPException(
            status_code=403,
            detail="You can only warm your own cache"
        )
    
    chain_cache = get_chain_cache()
    background_tasks.add_task(chain_cache.pre_warm_cache, target_username)
    
    return {
        "message": f"Started pre-warming cache for user {target_username}",
        "target_user": target_username,
        "requested_by": current_user.username
    }

@router.delete(
    "/cache/clear/{target_username}",
    response_model=Dict,
    summary="Clear cache for a user",
    description="Clear the RAG chain cache for a specific user"
)
async def clear_cache_for_user(
    target_username: str,
    current_user: User = Depends(get_current_user)
):
    """
    ## Clear User Cache
    
    Clear the RAG chain cache for a specific user.
    Users can only clear their own cache unless they have admin privileges.
    
    Requires a valid JWT token in the Authorization header.
    """
    # Allow users to clear their own cache, or admins to clear any cache
    if current_user.username != target_username and current_user.username != "admin":
        raise HTTPException(
            status_code=403,
            detail="You can only clear your own cache"
        )
    
    chain_cache = get_chain_cache()
    chain_cache.invalidate_user_cache(target_username)
    
    return {
        "message": f"Cleared cache for user {target_username}",
        "target_user": target_username,
        "requested_by": current_user.username
    }