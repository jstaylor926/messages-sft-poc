from fastapi import FastAPI
from backend.app.api import routes

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Email Style PoC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
