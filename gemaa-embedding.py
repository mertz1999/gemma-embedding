from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union, List
import uvicorn
import time
import numpy as np
import os
import sys 

app = FastAPI(
    title="Embedding Service",
    description="FastAPI service for generating embeddings using Google's Gemma model",
    version="1.0.0"
)

# Global variable to store the model
model = None

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("🚀 Embedding service startup complete")
    print("📡 Server is ready to accept requests")

def load_model():
    """Load the model lazily to optimize memory usage"""
    global model
    if model is None:
        try:
            print("🔄 Loading embedding model: google/embeddinggemma-300m")
            print("⏳ This may take a few moments on first request...")
            model = SentenceTransformer("google/embeddinggemma-300m")
            print("✅ Model loaded successfully and ready for inference")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return model

# Print startup message
print("🚀 Embedding service starting up...")
print("📦 Model will be loaded on first request")
print("✅ FastAPI app initialized")
print("🌐 Server ready to accept requests")

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Input text to get embeddings for")
    model: str = Field(default="text-embedding-ada-002", description="ID of the model to use")
    encoding_format: str = Field(default="float", description="The format to return the embeddings in")

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    permission: List[dict] = []
    root: str = ""
    parent: str = None

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

@app.get("/")
def root():
    """Root endpoint for basic connectivity check"""
    return {"message": "Embedding service is running", "status": "ok"}

@app.get("/health")
def health_check():
    """Health check endpoint - doesn't load model to keep it fast"""
    return {"status": "healthy", "service": "embedding-service"}

@app.get("/health/full")
def full_health_check():
    """Full health check that includes model loading"""
    try:
        # Try to load the model to ensure it's working
        load_model()
        return {"status": "healthy", "model": "embeddinggemma-300m", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "model_loaded": False}

@app.get("/model/status")
def model_status():
    """Check if the model is loaded without loading it"""
    global model
    return {
        "model_loaded": model is not None,
        "model_name": "google/embeddinggemma-300m",
        "status": "ready" if model is not None else "not_loaded"
    }

@app.post("/model/warmup")
def warmup_model():
    """Warm up the model by loading it"""
    try:
        load_model()
        return {"status": "success", "message": "Model warmed up successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to warm up model: {str(e)}"}

@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    """List available models - OpenAI API compatible endpoint"""
    current_time = int(time.time())
    
    models = [
        ModelInfo(
            id="text-embedding-ada-002",
            created=current_time,
            owned_by="local",
            permission=[],
            root="text-embedding-ada-002",
            parent=None
        ),
        ModelInfo(
            id="embeddinggemma-300m",
            created=current_time,
            owned_by="local", 
            permission=[],
            root="embeddinggemma-300m",
            parent=None
        )
    ]
    
    return ModelsResponse(object="list", data=models)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def get_embeddings(req: EmbeddingRequest):
    """Generate embeddings for input text - OpenAI API compatible endpoint"""
    try:
        # Convert input to list if it's a string
        if isinstance(req.input, str):
            input_texts = [req.input]
        else:
            input_texts = req.input
        
        # Validate input
        if not input_texts or len(input_texts) == 0:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Check individual document length (OpenAI has limits per document)
        for i, text in enumerate(input_texts):
            if len(text) > 8192:
                raise HTTPException(status_code=400, detail=f"Document {i} too long: {len(text)} characters (limit: 8192)")
        
        # Check total batch size (reasonable limit to prevent memory issues)
        total_chars = sum(len(text) for text in input_texts)
        if total_chars > 100000:  # Reasonable limit for batch processing
            raise HTTPException(status_code=400, detail=f"Batch too large: {total_chars} characters (limit: 100000)")
        
        # Load model and generate embeddings
        model_instance = load_model()
        vectors = model_instance.encode(input_texts, convert_to_numpy=True)
        
        # Create response data
        embedding_data = []
        for i, vector in enumerate(vectors):
            embedding_data.append(EmbeddingData(
                object="embedding",
                embedding=vector.tolist(),
                index=i
            ))
        
        # Calculate usage (rough approximation)
        prompt_tokens = total_chars // 4  # Rough token estimation
        total_tokens = prompt_tokens
        
        return EmbeddingResponse(
            object="list",
            data=embedding_data,
            model=req.model,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    try:
        # Get port from environment variable (required for Cloud Run)
        port = int(os.environ.get("PORT", 8080))
        print(f"🚀 Starting server on port {port}")
        print(f"🌍 Environment PORT: {os.environ.get('PORT', 'not set')}")
        print(f"📦 Working directory: {os.getcwd()}")
        print(f"🐍 Python version: {sys.version}")
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
