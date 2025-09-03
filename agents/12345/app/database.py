import os
import json
import sqlite3
import aiosqlite
import time
import random
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite connection configuration
SQLITE_DB_PATH = 'rag_chat_history.db'
SQLITE_TIMEOUT = 30.0  # Increase timeout to 30 seconds
MAX_RETRIES = 5
RETRY_DELAY_BASE = 0.1  # Base delay in seconds

def load_or_initialize_json(file_path):
    """Load JSON data from file or initialize if not exists."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize default structure
            default_data = {
                "files": {},
                "processed_files": {},
                "processing_status": {
                    "in_progress": [],
                    "completed": [],
                    "failed": []
                },
                "chat_history": {}
            }
            save_json_to_file(default_data, file_path)
            return default_data
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return {
            "files": {},
            "processed_files": {},
            "processing_status": {
                "in_progress": [],
                "completed": [],
                "failed": []
            },
            "chat_history": {}
        }

def save_json_to_file(data, file_path):
    """Save JSON data to file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")

def update_file_count(file_name, data):
    """Update file count in the JSON data."""
    if "files" not in data:
        data["files"] = {}
    
    if file_name in data["files"]:
        data["files"][file_name] += 1
    else:
        data["files"][file_name] = 1

def update_processing_status(file_name, status, data):
    """Update processing status for a file."""
    if status == "start":
        # Add to in_progress list
        if file_name not in data["processing_status"]["in_progress"]:
            data["processing_status"]["in_progress"].append(file_name)
    elif status == "complete":
        # Move from in_progress to completed
        if file_name in data["processing_status"]["in_progress"]:
            data["processing_status"]["in_progress"].remove(file_name)
        if file_name not in data["processing_status"]["completed"]:
            data["processing_status"]["completed"].append(file_name)
        
        # Add to processed_files with timestamp
        if "processed_files" not in data:
            data["processed_files"] = {}
        data["processed_files"][file_name] = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
    elif status == "fail":
        # Move from in_progress to failed
        if file_name in data["processing_status"]["in_progress"]:
            data["processing_status"]["in_progress"].remove(file_name)
        if file_name not in data["processing_status"]["failed"]:
            data["processing_status"]["failed"].append(file_name)
        
        # Add to processed_files with error status
        if "processed_files" not in data:
            data["processed_files"] = {}
        data["processed_files"][file_name] = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }

def update_file_processing_timestamp(file_name, data):
    """
    Update or add a processing timestamp for a file.
    This is separate from status updates and ensures we have
    a record of when the file was last processed.
    """
    if "processed_files" not in data:
        data["processed_files"] = {}
    
    # Add or update timestamp for the file
    if file_name not in data["processed_files"]:
        data["processed_files"][file_name] = {}
    
    data["processed_files"][file_name]["last_processed"] = datetime.now().isoformat()

def fetch_session_chat_history(session_id, data):
    """Retrieve chat history for a session."""
    if "chat_history" not in data:
        data["chat_history"] = {}
    
    if session_id not in data["chat_history"]:
        data["chat_history"][session_id] = []
    
    return data["chat_history"][session_id]

def store_chat_history(question: str, answer: str, data: dict, username: str, session_id: str) -> None:
    """
    Store chat history in the JSON data structure.
    
    Args:
        question: The user's question
        answer: The AI's answer
        data: JSON data structure to update 
        username: Username for this chat
        session_id: Session ID for this chat
    """
    timestamp = datetime.now().isoformat()
    
    # Create chat entry
    chat_entry = {
        "question": question,
        "answer": answer,
        "timestamp": timestamp
    }
    
    # Initialize chat_history if not exists
    if "chat_history" not in data:
        data["chat_history"] = {}
    
    # Initialize session chat history
    if session_id not in data["chat_history"]:
        data["chat_history"][session_id] = []
    
    # Add entry to session chat history
    data["chat_history"][session_id].append(chat_entry)
    
    # Initialize users structure if needed
    if "users" not in data:
        data["users"] = {}
    if username not in data["users"]:
        data["users"][username] = {"sessions": [], "files": [], "chat_history": {}}
    
    # Initialize user's chat history for this session
    if "chat_history" not in data["users"][username]:
        data["users"][username]["chat_history"] = {}
    if session_id not in data["users"][username]["chat_history"]:
        data["users"][username]["chat_history"][session_id] = []
    
    # Ensure session is tracked for user
    if "sessions" not in data["users"][username]:
        data["users"][username]["sessions"] = []
    if session_id not in data["users"][username]["sessions"]:
        data["users"][username]["sessions"].append(session_id)
    
    # Add entry to user's session chat history
    data["users"][username]["chat_history"][session_id].append(chat_entry)
    
    # No need to save to file here - caller is responsible for that
    return

def init_db():
    """Initialize database with WAL journal mode for better concurrency"""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT)
        cursor = conn.cursor()
        
        # Set journal mode to WAL for better concurrency support
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        # Configure other SQLite parameters for better concurrency
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        cursor.execute("PRAGMA busy_timeout=30000;")  # 30 seconds in milliseconds
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            username TEXT,
            session_id TEXT,
            timestamp TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with WAL journal mode")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

async def insert_chat_data_async(question, answer, username, session_id):
    """Insert chat data with retry logic for database locking issues"""
    retries = 0
    last_error = None
    
    while retries < MAX_RETRIES:
        try:
            async with aiosqlite.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT) as conn:
                cursor = await conn.cursor()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                await cursor.execute('''
                INSERT INTO chat_history (question, answer, username, session_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
                ''', (question, answer, username, session_id, timestamp))

                await conn.commit()
                return  # Success, exit the retry loop
                
        except aiosqlite.OperationalError as e:
            if "database is locked" in str(e):
                retries += 1
                last_error = e
                
                # Calculate delay with exponential backoff and jitter
                delay = min(RETRY_DELAY_BASE * (2 ** retries) + random.uniform(0, 0.1), 5.0)
                
                logger.warning(f"Database locked, retrying in {delay:.2f}s (attempt {retries}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
            else:
                # Not a locking error, re-raise
                raise
        except Exception as e:
            logger.error(f"Error inserting chat data: {str(e)}")
            raise
    
    # If we get here, we've exhausted our retries
    logger.error(f"Failed to insert chat data after {MAX_RETRIES} retries")
    raise last_error or Exception("Failed to insert data into database after multiple retries")

async def fetch_chat_history_async(username):
    """Fetch chat history with retry logic"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with aiosqlite.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT) as conn:
                cursor = await conn.execute('''
                    SELECT session_id, question, answer, timestamp FROM chat_history
                    WHERE username = ?
                    ORDER BY session_id, timestamp
                ''', (username,))
                session_history = {}
                
                async for row in cursor:
                    session_id = row[0]
                    question = row[1]
                    answer = row[2]
                    timestamp = row[3]
                    
                    if session_id not in session_history:
                        session_history[session_id] = []
                    
                    session_history[session_id].append({"question": question, "answer": answer, "timestamp": timestamp})
                
                await cursor.close()
                return session_history
                
        except aiosqlite.OperationalError as e:
            if "database is locked" in str(e):
                retries += 1
                
                if retries >= MAX_RETRIES:
                    logger.error(f"Failed to fetch chat history after {MAX_RETRIES} retries")
                    raise
                
                delay = min(RETRY_DELAY_BASE * (2 ** retries) + random.uniform(0, 0.1), 5.0)
                logger.warning(f"Database locked during fetch, retrying in {delay:.2f}s (attempt {retries}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            logger.error(f"Error fetching chat history: {str(e)}")
            raise

async def fetch_session_chat_history(username, session_id):
    """Fetch session chat history with retry logic"""
    chat_history = []
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            async with aiosqlite.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT) as conn:
                cursor = await conn.execute('''
                    SELECT question, answer FROM chat_history
                    WHERE username = ? AND session_id = ?
                    ORDER BY timestamp
                ''', (username, session_id))
                
                async for row in cursor:
                    question = row[0]
                    answer = row[1]
                    chat_history.append(HumanMessage(content=question))
                    chat_history.append(AIMessage(content=answer))
                
                await cursor.close()
                return chat_history
                
        except aiosqlite.OperationalError as e:
            if "database is locked" in str(e):
                retries += 1
                
                if retries >= MAX_RETRIES:
                    logger.error(f"Failed to fetch session chat history after {MAX_RETRIES} retries")
                    raise
                
                delay = min(RETRY_DELAY_BASE * (2 ** retries) + random.uniform(0, 0.1), 5.0)
                logger.warning(f"Database locked during session fetch, retrying in {delay:.2f}s (attempt {retries}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            logger.error(f"Error fetching session chat history: {str(e)}")
            raise

async def clear_history_async(username, session_id=None):
    """Clear chat history with retry logic"""
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            async with aiosqlite.connect(SQLITE_DB_PATH, timeout=SQLITE_TIMEOUT) as conn:
                cursor = await conn.cursor()
                
                if session_id:
                    # Clear specific session
                    await cursor.execute(
                        "DELETE FROM chat_history WHERE username = ? AND session_id = ?", 
                        (username, session_id)
                    )
                else:
                    # Clear all sessions for user
                    await cursor.execute(
                        "DELETE FROM chat_history WHERE username = ?", 
                        (username,)
                    )
                
                await conn.commit()
                return {"status": "success", "message": "Chat history cleared"}
                
        except aiosqlite.OperationalError as e:
            if "database is locked" in str(e):
                retries += 1
                
                if retries >= MAX_RETRIES:
                    logger.error(f"Failed to clear history after {MAX_RETRIES} retries")
                    raise
                
                delay = min(RETRY_DELAY_BASE * (2 ** retries) + random.uniform(0, 0.1), 5.0)
                logger.warning(f"Database locked during clear operation, retrying in {delay:.2f}s (attempt {retries}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            raise
