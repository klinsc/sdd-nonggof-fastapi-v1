import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging import request_id_ctx
from app.routers import chat, stats

settings = get_settings()

app = FastAPI(
    title="น้องกอฟ — PEA SDD AI Assistant",
    lifespan=lifespan,
    # Don't expose interactive docs / OpenAPI schema publicly in production;
    # enable them only when DEBUG is on.
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


app.include_router(chat.router)
app.include_router(stats.router)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


@app.get("/healthz")
async def healthz(request: Request):
    ready = bool(getattr(request.app.state, "ready", False))
    return {"status": "ok" if ready else "starting", "ready": ready}
