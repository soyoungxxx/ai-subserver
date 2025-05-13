from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from models.yolo.yolo_model import YOLOModel
from schemas.yolo_schemas import Polygon
import shutil
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# yolo 일반 추론
@app.post("/predict/yolo")
async def predict_yolo(
    file: UploadFile = File(...),
    polygon_json: Optional[str] = Form(None),
):
    # input / output 경로 지정
    input_path = f"temp/input_{file.filename}";
    output_path = f"temp/output_{file.filename}";

    # 디렉토리 없으면 생성
    os.makedirs("temp", exist_ok=True)

    # 파일 저장
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer);
    
    # json 파싱
    polygon_data = Polygon.parse_raw(polygon_json);

    # 모델 호출
    model = YOLOModel();
    model.predict(input_path, output_path, polygon_data);

    # output 디렉토리 내 결과 WAS로 전송
    return {"output": output_path, "data": polygon_data};