import os
import json
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SECRET_KEY = "GkP8pF3vZtW5Qy7XdKjM2sL9bN4rH6cE1aA0zR8xV3wJ"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120  # Longer expiration since no password security
USER_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user.json")

# Use API key header instead of OAuth2 - simpler approach
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    expires_at: str
    session_id: str

class UsernameRequest(BaseModel):
    username: str

class TokenData(BaseModel):
    username: Optional[str] = None
    session_id: Optional[str] = None

class User(BaseModel):
    username: str
    session_id: str

def load_users() -> Dict[str, str]:
    """Load users from the JSON file"""
    try:
        if os.path.exists(USER_FILE_PATH):
            with open(USER_FILE_PATH, 'r') as f:
                return json.load(f)
        else:
            # Create default user file if it doesn't exist
            default_users = {"admin": "admin123", "user1": "password1", "testuser": "testpass"}
            os.makedirs(os.path.dirname(USER_FILE_PATH), exist_ok=True)
            with open(USER_FILE_PATH, 'w') as f:
                json.dump(default_users, f, indent=2)
            logger.info(f"Created default user file at {USER_FILE_PATH}")
            return default_users
    except Exception as e:
        logger.error(f"Error loading user file: {e}")
        return {"admin": "admin123"}  # Default fallback

def user_exists(username: str) -> bool:
    """Check if a user exists in the user file without validating password"""
    users = load_users()
    return username in users

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT token with expiration time"""
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"Created token for user {data.get('sub')} with expiry {expire}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise

def create_session_id() -> str:
    """Create a unique session ID"""
    return str(uuid.uuid4())

async def get_current_user(authorization: str = Depends(API_KEY_HEADER)) -> User:
    """Dependency to get the current user from a JWT token in Authorization header"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not authorization:
        logger.warning("No authorization header provided")
        raise credentials_exception
        
    # Format should be "Bearer {token}"
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        logger.warning(f"Invalid authentication scheme: {scheme}")
        raise credentials_exception
    
    if not token:
        logger.warning("No token provided")
        raise credentials_exception
        
    try:
        # Decode and verify the JWT token using the secret key
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        session_id: str = payload.get("session_id")
        
        if username is None:
            logger.warning("Username missing from token")
            raise credentials_exception
            
        if session_id is None:
            logger.warning("Session ID missing from token")
            raise credentials_exception
            
        token_data = TokenData(username=username, session_id=session_id)
        logger.debug(f"Token validated for user {username} with session {session_id}")
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise credentials_exception
    
    # Verify user exists without password check
    if not user_exists(token_data.username):
        logger.warning(f"User {token_data.username} from token does not exist")
        raise credentials_exception
    
    return User(username=token_data.username, session_id=token_data.session_id)

def verify_token(token: str) -> Optional[Dict]:
    """Utility function to verify a token without HTTP dependencies"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"Token verification failed: {str(e)}")
        return None
