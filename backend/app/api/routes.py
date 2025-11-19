from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class GenerateRequest(BaseModel):
    subject: str
    body: str
    mode: str = "lora"  # lora, gpt4, hybrid

class GenerateResponse(BaseModel):
    lora_reply: Optional[str] = None
    gpt4_reply: Optional[str] = None
    hybrid_reply: Optional[str] = None
    used_mode: str

from backend.app.services.inference import InferenceService

inference_service = InferenceService()

@router.post("/generate", response_model=GenerateResponse)
async def generate_reply(request: GenerateRequest):
    lora_reply = None
    gpt4_reply = None
    hybrid_reply = None

    if request.mode == "lora":
        lora_reply = inference_service.generate_lora(request.subject, request.body)
    elif request.mode == "gpt4":
        gpt4_reply = inference_service.generate_gpt4(request.subject, request.body)
    elif request.mode == "hybrid":
        hybrid_reply = inference_service.generate_hybrid(request.subject, request.body)

    return GenerateResponse(
        lora_reply=lora_reply,
        gpt4_reply=gpt4_reply,
        hybrid_reply=hybrid_reply,
        used_mode=request.mode
    )

class ScoreRequest(BaseModel):
    incoming_subject: str
    incoming_body: str
    candidate_reply: str

class ScoreResponse(BaseModel):
    content_bleu: float
    content_bertscore: float
    fluency_cola: float

@router.post("/score", response_model=ScoreResponse)
async def score_reply(request: ScoreRequest):
    # Placeholder for scoring logic
    return ScoreResponse(
        content_bleu=0.8,
        content_bertscore=0.9,
        fluency_cola=0.95
    )
