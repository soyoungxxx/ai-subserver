from fastapi import APIRouter, UploadFile, File
from services.aqua_service import handle_aqua_detection

router = APIRouter()

@router.post("")
async def detect_aqua(
    file: UploadFile = File(...),
):
    result = await handle_aqua_detection(file);  
    return result;
