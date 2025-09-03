from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import LOCAL_MODEL_PATH,GOOGLE_API_KEY
import logging
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.)
)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings():
    """Get embedding model instance."""
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    return HuggingFaceInstructEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # return GoogleGenerativeAIEmbeddings(
    #         google_api_key=GOOGLE_API_KEY,
    #         model="models/embedding-001"
    #     )



def get_llm():
    """Get language model with retry settings."""
    logger.info("Initializing LLM with retry settings")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=1024,
        # Add retry configuration
        max_retries=4,
        retry_min_seconds=1,
        retry_max_seconds=30
    )

def get_vision_llm():
    """Get vision capable LLM with retry settings."""
    logger.info("Initializing vision LLM with retry settings")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Updated from gemini-pro-vision
        temperature=0.3,
        max_tokens=1024,
        # Add retry configuration
        max_retries=4,
        retry_min_seconds=1,
        retry_max_seconds=30,rate_limiter=rate_limiter
    ).with_retry(stop_after_attempt=4)
