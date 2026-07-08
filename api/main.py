"""
NBA Agentic AI Trade Simulator — FastAPI Application.

Production-quality multi-agent AI system with:
- 10 specialized agents (GM, Trade, Salary, Championship, Injury, Chemistry, etc.)
- GPT function calling (never answers directly — always calls tools first)
- PostgreSQL persistence, Redis caching, ChromaDB RAG
- Prometheus metrics, structured logging, request tracing
- Background task execution for heavy Monte Carlo simulations

Endpoints:
    POST /ask           — Natural language query
    POST /trade         — Structured trade analysis
    POST /optimize      — Roster optimization
    POST /championship  — Championship odds simulation
    POST /roster        — Roster chemistry analysis
    POST /draft         — Draft recommendation
    POST /freeagency    — Free agency recommendation
    POST /compare       — Player comparison
    GET  /health        — Health check
    GET  /metrics       — Prometheus metrics

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from observability.logging import setup_logging, get_logger
from observability.tracing import TracingMiddleware

# Configure structured logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("app_startup", message="Initializing NBA Agentic AI System...")

    # Pre-load data and models
    try:
        from nba_analysis.data_loader import get_player_stats_with_predictions
        logger.info("loading_data", message="Loading player stats and ML models...")
        get_player_stats_with_predictions()
        logger.info("data_loaded", message="Player data and models ready")
    except Exception as e:
        logger.error("data_load_error", error=str(e))

    # Initialize RAG pipeline
    try:
        from rag.ingest import ingest_documents
        chunks = ingest_documents()
        logger.info("rag_initialized", chunks=chunks)
    except Exception as e:
        logger.warning("rag_init_skip", error=str(e))

    # Initialize database (create tables if needed)
    try:
        from db.engine import init_db
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("db_init_skip", error=str(e))

    logger.info("app_ready", message="NBA Agentic AI System is operational")
    yield

    # Shutdown
    logger.info("app_shutdown", message="Shutting down...")
    try:
        from db.engine import close_db
        await close_db()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="NBA Agentic AI Trade Simulator",
        description=(
            "Production-quality multi-agent AI system for NBA trade analysis. "
            "10 specialized agents use GPT function calling to reason, plan, "
            "call tools, combine outputs, and generate explainable recommendations. "
            "GPT never directly answers — it always calls tools first."
        ),
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request tracing
    app.add_middleware(TracingMiddleware)

    # Register routes
    from api.routes.ask import router as ask_router
    from api.routes.trade import router as trade_router
    from api.routes.optimize import router as optimize_router
    from api.routes.championship import router as championship_router
    from api.routes.roster import router as roster_router
    from api.routes.draft import router as draft_router
    from api.routes.freeagency import router as freeagency_router
    from api.routes.compare import router as compare_router
    from api.routes.health import router as health_router

    app.include_router(ask_router)
    app.include_router(trade_router)
    app.include_router(optimize_router)
    app.include_router(championship_router)
    app.include_router(roster_router)
    app.include_router(draft_router)
    app.include_router(freeagency_router)
    app.include_router(compare_router)
    app.include_router(health_router)

    @app.get("/")
    async def root():
        return {
            "service": "NBA Agentic AI Trade Simulator",
            "version": "3.0.0",
            "architecture": "Multi-Agent with GPT Function Calling",
            "agents": [
                "GeneralManagerAgent",
                "TradeAnalysisAgent",
                "SalaryCapAgent",
                "ChampionshipSimAgent",
                "InjuryRiskAgent",
                "ChemistryAgent",
                "RosterOptimizerAgent",
                "FreeAgencyAgent",
                "DraftAgent",
                "ReportGeneratorAgent",
            ],
            "endpoints": [
                "POST /ask",
                "POST /trade",
                "POST /optimize",
                "POST /championship",
                "POST /roster",
                "POST /draft",
                "POST /freeagency",
                "POST /compare",
                "GET /health",
                "GET /metrics",
            ],
        }

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
