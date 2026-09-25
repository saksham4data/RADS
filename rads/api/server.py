import asyncio
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from rads.api.schemas import ConfigResponse, EventResponse, HealthResponse


def create_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        app.state.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.clients = set()
    app.state.loop = None

    @app.get("/health", response_model=HealthResponse)
    def health():
        return engine.health.get_health()

    @app.get("/config", response_model=ConfigResponse)
    def config():
        return engine.config.config

    @app.get("/events", response_model=list[EventResponse])
    def events():
        return engine.lifecycle.get_recent_events(engine.config.api_event_buffer_size)

    @app.get("/events/latest", response_model=EventResponse)
    def latest_event():
        recent = engine.lifecycle.get_recent_events(engine.config.api_event_buffer_size)
        if not recent:
            raise HTTPException(status_code=404, detail="event not found")
        return recent[-1]

    @app.get("/events/{event_id}", response_model=EventResponse)
    def event_by_id(event_id: str):
        for event in engine.lifecycle.get_recent_events(engine.config.api_event_buffer_size):
            if event["event_id"] == event_id:
                return event
        raise HTTPException(status_code=404, detail="event not found")

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        app.state.clients.add(websocket)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, WebSocketDisconnected):
            pass
        finally:
            app.state.clients.discard(websocket)

    def _broadcast(event):
        loop = app.state.loop
        if loop is None or loop.is_closed():
            return
        for ws in list(app.state.clients):
            asyncio.run_coroutine_threadsafe(_send(ws, event), loop)

    async def _send(ws, event):
        try:
            await ws.send_json(event)
        except Exception:
            app.state.clients.discard(ws)

    engine.register_handler(_broadcast)
    return app


def start_api(engine):
    app = create_app(engine)
    config = uvicorn.Config(
        app,
        host=engine.config.api_host,
        port=engine.config.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, name="rads-api", daemon=True)
    thread.start()
    return server
