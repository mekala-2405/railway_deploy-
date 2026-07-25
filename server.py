"""Punch.io — FastAPI backend for the React frontend.

Handles Discord sync, data export, and RAG Q&A.
Run:  uv run uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
import re
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent / "data"
EXPORT_DIR = Path(__file__).parent / "frontend" / "public" / "data"


class OnboardRequest(BaseModel):
    discord_token: str
    guild_id: str = ""
    groq_key: str = ""


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/onboard")
def onboard(req: OnboardRequest):
    """Sync Discord channels and export data for the frontend."""
    os.environ["DISCORD_BOT_TOKEN"] = req.discord_token
    if req.guild_id:
        os.environ["DISCORD_GUILD_ID"] = req.guild_id
    if req.groq_key:
        os.environ["GROQ_API_KEY"] = req.groq_key

    DATA_DIR.mkdir(exist_ok=True)
    db_path = str(DATA_DIR / "punch.db")
    faiss_dir = str(DATA_DIR / "faiss_db")

    # Fresh start on every onboard: the DB and FAISS index accumulate across syncs
    # (upsert only appends), so a new bot would otherwise show the previous bot's
    # messages too. Clear both before syncing.
    import shutil
    Path(db_path).unlink(missing_ok=True)
    shutil.rmtree(faiss_dir, ignore_errors=True)

    from core import store as _store
    _store.init_db(db_path)

    # Sync all discoverable channels
    try:
        from sync import discover_discord_connectors
        from ingest.pipeline import sync_source
        connectors = discover_discord_connectors(req.discord_token)
        results = {}
        for c in connectors:
            try:
                r = sync_source(c, db_path=db_path, faiss_dir=faiss_dir)
                results[c.source_id] = r
            except Exception as e:
                results[getattr(c, "source_id", c.name)] = {"error": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")

    # Export data for the frontend
    try:
        from export_site import extract_timeline, message_to_dict
        from core import store
        messages = store.get_messages(db_path=db_path)
        if not messages:
            raise HTTPException(status_code=400, detail="No messages synced.")

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        (EXPORT_DIR / "messages.json").write_text(
            json.dumps([message_to_dict(m) for m in messages], ensure_ascii=False, indent=2))

        channels = sorted({m.channel for m in messages})
        projects = sorted({m.project for m in messages if m.project})
        meta = {"message_count": len(messages), "channels": channels, "projects": projects,
                "date_range": [messages[0].timestamp[:10], messages[-1].timestamp[:10]]}
        (EXPORT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        events = extract_timeline(messages)
        (EXPORT_DIR / "timeline.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    return {"synced": len(messages), "events": len(events), "channels": channels}


@app.post("/api/ask")
def ask(req: AskRequest):
    """RAG Q&A over synced messages."""
    db_path = str(DATA_DIR / "punch.db")
    faiss_dir = str(DATA_DIR / "faiss_db")
    try:
        from ingest.pipeline import embed_new
        embed_new(db_path=db_path, faiss_dir=faiss_dir)
        from generation import ask_question, retrieve_context
        docs, _ = retrieve_context(req.question)
        answer = ask_question(req.question, history=req.history)
        sources = [{"author": d.metadata.get("author"), "channel": d.metadata.get("channel"),
                    "timestamp": d.metadata.get("timestamp"), "content": d.page_content} for d in docs]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve exported data files — create dir so mount always works
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(EXPORT_DIR)), name="data")

# Serve React build (must be last)
DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="static")
