import os
from dotenv import load_dotenv
import langfuse
from langfuse import trace
from google.adk.agents import *
# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
API_URL = os.getenv('API_URL')

# File paths
INDEX_FILE_PATH = os.getenv('index_file_path')
INDEX_PATH = os.getenv('index_path')
LOCAL_MODEL_PATH = os.getenv('local_model_path')
IMAGE_OUTPUT_PATH = os.getenv('image_output_path')
JSON_FILE_PATH = os.path.join(DATA_DIR, "document_status.json")
JSON_USERS_DATA = os.getenv('json_users_data')
SEMANTIC_CACHE = os.getenv('semantic_cache')

# Set environment variables globally
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Embeddings config
MODEL_KWARGS = {'device': 'cpu'}
ENCODE_KWARGS = {'normalize_embeddings': True}

# Headers for HuggingFace API
HEADERS = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# Other configuration settings can be added here
DEBUG = True

def create_directories():
    """Create necessary directories if they don't exist"""
    paths = [
        os.path.dirname(JSON_FILE_PATH) if JSON_FILE_PATH else None,
        IMAGE_OUTPUT_PATH,
        INDEX_PATH,
        os.path.dirname(SEMANTIC_CACHE) if SEMANTIC_CACHE else None
    ]
    
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)
