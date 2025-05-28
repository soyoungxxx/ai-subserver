from fastapi import APIRouter, UploadFile, File
from services.yolo_service import handle_yolo_detection

router = APIRouter()

@router.post("")
async def detect_yolo(
    file: UploadFile = File(...),
):
    result = await handle_yolo_detection(file);  
    return result;
