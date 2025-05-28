from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from services.mix_service import handle_mix_detection

router = APIRouter()

@router.post("")
async def detect_aqua(
    file: UploadFile = File(...),
    polygon_json: Optional[str] = Form(None),
):
    result = await handle_mix_detection(file, polygon_json);  
    return result;
