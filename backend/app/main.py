"""
Backend API for Legal KG Visualization
=======================================
FastAPI server connecting to Neo4j KG + LLM inference.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import graph, search, qa, stats, document

app = FastAPI(
    title="Legal KG Visualization API",
    description="API for exploring Indonesian Legal Knowledge Graph",
    version="0.1.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(qa.router, prefix="/api", tags=["qa"])
app.include_router(stats.router, prefix="/api", tags=["stats"])
app.include_router(document.router, prefix="/api", tags=["document"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
