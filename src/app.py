# -*- coding: utf-8 -*-
"""
PhenoTagger FastAPI Application
Simple REST API for HPO term annotation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import logging
import time
from contextlib import asynccontextmanager

# Import our PhenoTagger API
from phenotagger_api import PhenoTaggerAPI, create_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global API instance
phenotagger_api: Optional[PhenoTaggerAPI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the PhenoTagger API"""
    global phenotagger_api
    
    # Startup
    logger.info("Starting PhenoTagger API...")
    try:
        phenotagger_api = create_api()
        logger.info("PhenoTagger API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PhenoTagger API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PhenoTagger API...")
    phenotagger_api = None

# Create FastAPI app
app = FastAPI(
    title="PhenoTagger API",
    description="Human Phenotype Ontology (HPO) term annotation service",
    version="2.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnnotateRequest(BaseModel):
    text: str = Field(..., description="Text to annotate with HPO terms", min_length=1)
    only_longest: Optional[bool] = Field(False, description="Return only longest overlapping concepts")
    abbr_recognition: Optional[bool] = Field(True, description="Enable abbreviation recognition")
    threshold: Optional[float] = Field(0.95, description="ML model threshold (0.0-1.0)", ge=0.0, le=1.0)

class BatchAnnotateRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to annotate", min_items=1)
    only_longest: Optional[bool] = Field(False, description="Return only longest overlapping concepts")
    abbr_recognition: Optional[bool] = Field(True, description="Enable abbreviation recognition")
    threshold: Optional[float] = Field(0.95, description="ML model threshold (0.0-1.0)", ge=0.0, le=1.0)

class AnnotateResponse(BaseModel):
    text: str
    hpo_terms: str
    hpo_ids: str = Field(..., description="HPO IDs only: HP:xxxxx;HP:xxxxx")
    processing_time: float

class BatchAnnotateResponse(BaseModel):
    results: List[AnnotateResponse]
    total_processing_time: float
    count: int

class HealthResponse(BaseModel):
    status: str
    model_info: Dict
    timestamp: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "PhenoTagger API",
        "version": "2.0",
        "description": "HPO term annotation service",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if phenotagger_api is None:
        raise HTTPException(status_code=503, detail="PhenoTagger API not initialized")
    
    try:
        model_info = phenotagger_api.get_model_info()
        return HealthResponse(
            status="healthy",
            model_info=model_info,
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/annotate", response_model=AnnotateResponse)
async def annotate_text(request: AnnotateRequest):
    """
    Annotate a single text with HPO terms
    
    Returns HPO terms in format: "term1 (HP:xxxxx);term2 (HP:xxxxx);..."
    """
    if phenotagger_api is None:
        raise HTTPException(status_code=503, detail="PhenoTagger API not initialized")
    
    try:
        start_time = time.time()
        
        # Call annotation API
        result = phenotagger_api.annotate_text(
            text=request.text,
            only_longest=request.only_longest,
            abbr_recognition=request.abbr_recognition,
            threshold=request.threshold
        )
        
        processing_time = time.time() - start_time
        
        return AnnotateResponse(
            text=request.text,
            hpo_terms=result[request.text],
            hpo_ids=result["hpo_ids"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

@app.post("/annotate/batch", response_model=BatchAnnotateResponse)
async def annotate_batch(request: BatchAnnotateRequest):
    """
    Annotate multiple texts with HPO terms
    
    No limit on the number of texts (removed 100 text limit)
    """
    if phenotagger_api is None:
        raise HTTPException(status_code=503, detail="PhenoTagger API not initialized")
    
    # Optional: Add warning for very large batches
    if len(request.texts) > 1000:
        logger.warning(f"Processing large batch of {len(request.texts)} texts. This may take a while.")
    
    try:
        start_time = time.time()
        
        # Call batch annotation API
        batch_results = phenotagger_api.annotate_batch(
            texts=request.texts,
            only_longest=request.only_longest,
            abbr_recognition=request.abbr_recognition,
            threshold=request.threshold
        )
        
        total_processing_time = time.time() - start_time
        
        # Format results
        results = []
        for text in request.texts:
            annotation_info = batch_results.get(text, {"hpo_terms": "", "hpo_ids": ""})
            results.append(AnnotateResponse(
                text=text,
                hpo_terms=annotation_info["hpo_terms"],
                hpo_ids=annotation_info["hpo_ids"],
                processing_time=total_processing_time / len(request.texts)  # Average time
            ))
        
        return BatchAnnotateResponse(
            results=results,
            total_processing_time=total_processing_time,
            count=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch annotation failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current API configuration"""
    if phenotagger_api is None:
        raise HTTPException(status_code=503, detail="PhenoTagger API not initialized")
    
    try:
        model_info = phenotagger_api.get_model_info()
        env_info = phenotagger_api.config.get_env_info()
        
        return {
            "model_info": model_info,
            "environment": env_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )