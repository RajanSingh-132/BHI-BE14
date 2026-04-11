"""
FastAPI application entry point.

Startup tasks:
  1. Attach MongoDB singleton to app.state.mongo.
  2. Create TTL index on session_state.updated_at (idempotent, 24 h expiry).
     This ensures stale sessions are automatically purged and do not
     survive server restarts to cause phantom dataset lookups.

No app.state.ACTIVE_DATASET or ACTIVE_DATASETS globals — session state
is keyed by the session_id UUID from the X-Session-ID request header.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.chat_routes import router as chat_router
from routes.upload import router as upload_router
from mongo_client import mongo_client as mongo
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    app.state.mongo = mongo
    mongo.ensure_ttl_index()
    yield
    # --- shutdown ---
    mongo.close()


app = FastAPI(title="AI Chatbot with MongoDB", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")


@app.get("/")
def home():
    return {"message": "AI Chatbot Running 🚀"}
