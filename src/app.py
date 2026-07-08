import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add current folder to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_answer

app = FastAPI(
    title="Industrial Knowledge Intelligence Center API",
    description="Backend API for querying industrial document vectors and generating answers via Gemini."
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
def query_rag_pipeline(request: QueryRequest):
    """
    POST API Endpoint
    Receives user question, runs the ChromaDB retrieval, and asks Gemini to synthesize.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        result = generate_answer(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setup Static Files Mounting
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount the static directory to serve index.html at root "/"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Configure UTF-8 logs on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
        
    print("=" * 60)
    print("Industrial RAG FastAPI Server Launching")
    print("Available locally at: http://localhost:8000")
    print("=" * 60)
    
    # Run server on port 8000 with auto-reload enabled
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
