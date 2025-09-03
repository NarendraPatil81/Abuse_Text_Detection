# RAG FastAPI Application with Authentication

A Retrieval-Augmented Generation (RAG) API built with FastAPI that provides document processing, vector search, and chat capabilities with JWT-based authentication.

## Features

- 🔐 **JWT Authentication** - Simple username-based authentication system
- 📄 **Document Upload & Processing** - Support for PDF, TXT, and DOCX files
- 🔍 **Vector Search** - Semantic search using embeddings
- 💬 **Chat Interface** - Interactive chat with uploaded documents
- 🚀 **FastAPI** - High-performance async API framework
- 📊 **Automatic Documentation** - Interactive API docs with Swagger UI

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /c:/Users/narendrakumar.patil/Downloads/RAG_FASTAPI/New_Fast
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn python-multipart python-jose[cryptography] passlib[bcrypt] langchain langchain-community chromadb sentence-transformers python-docx PyPDF2
   ```

4. **Run the application:**
   ```bash
   python -m app.main
   ```

5. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - API Base URL: http://localhost:8000
   - Health Check: http://localhost:8000/health

## Authentication

### Getting a Token

1. **Login with username:**
   ```bash
   curl -X POST "http://localhost:8000/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username": "admin"}'
   ```

2. **Response:**
   ```json
   {
     "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "token_type": "bearer"
   }
   ```

3. **Use token in requests:**
   ```bash
   curl -X GET "http://localhost:8000/api/documents/" \
        -H "Authorization: Bearer YOUR_TOKEN_HERE"
   ```

## API Endpoints

### Authentication
- `POST /auth/login` - Login with username to get JWT token

### Documents
- `GET /api/documents/` - List all uploaded documents
- `POST /api/documents/upload` - Upload a new document
- `DELETE /api/documents/{document_id}` - Delete a document
- `GET /api/documents/{document_id}/search` - Search within a specific document

### Chat
- `POST /api/chat/` - Send a message and get AI response
- `GET /api/chat/history` - Get chat history

### System
- `GET /` - API information and usage guide
- `GET /health` - Health check endpoint

## Usage Examples

### 1. Complete Workflow Example

```bash
# 1. Login to get token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin"}' | jq -r '.access_token')

# 2. Upload a document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@your_document.pdf"

# 3. List documents
curl -X GET "http://localhost:8000/api/documents/" \
  -H "Authorization: Bearer $TOKEN"

# 4. Chat about the document
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}'
```

### 2. Using Python Requests

```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/auth/login",
    json={"username": "admin"}
)
token = response.json()["access_token"]

# Headers for authenticated requests
headers = {"Authorization": f"Bearer {token}"}

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/documents/upload",
        headers=headers,
        files=files
    )

# Chat
chat_response = requests.post(
    "http://localhost:8000/api/chat/",
    headers=headers,
    json={"message": "Summarize the uploaded document"}
)
print(chat_response.json())
```

### 3. JavaScript/Frontend Example

```javascript
// Login and get token
const loginResponse = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'admin' })
});
const { access_token } = await loginResponse.json();

// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('http://localhost:8000/api/documents/upload', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${access_token}` },
  body: formData
});

// Chat
const chatResponse = await fetch('http://localhost:8000/api/chat/', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ message: 'What are the key points?' })
});
```

## Project Structure

```
New_Fast/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # Database configuration
│   ├── auth.py              # Authentication utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py          # User models
│   │   ├── document.py      # Document models
│   │   └── chat.py          # Chat models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication routes
│   │   ├── document.py      # Document management routes
│   │   └── chat.py          # Chat routes
│   └── services/
│       ├── __init__.py
│       ├── document_service.py    # Document processing
│       └── rag_service.py         # RAG implementation
├── data/                    # Data storage directory
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Supported File Types

- **PDF** (.pdf) - Portable Document Format
- **Text** (.txt) - Plain text files
- **Word** (.docx) - Microsoft Word documents

## Configuration

The application uses default configurations that work out of the box:

- **Host:** 0.0.0.0
- **Port:** 8000
- **JWT Secret:** Automatically generated
- **Token Expiration:** 30 minutes
- **CORS:** Enabled for all origins (development mode)

## Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables (Optional)

Create a `.env` file for custom configuration:

```env
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   pip install --upgrade fastapi uvicorn python-multipart
   ```

2. **File Upload Issues:**
   - Check file size limits
   - Ensure file type is supported
   - Verify authentication token

3. **Authentication Errors:**
   - Ensure token is included in Authorization header
   - Check token format: `Bearer <token>`
   - Verify token hasn't expired

### Checking Logs

The application logs important information to the console. Look for:
- Startup messages
- Authentication attempts
- File processing status
- Error messages

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation with Swagger UI, where you can:
- Test all endpoints
- View request/response schemas
- See authentication requirements
- Try out the API directly from the browser

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the interactive API documentation at `/docs`
2. Review this README file
3. Check the application logs for error messages
4. Ensure all dependencies are properly installed