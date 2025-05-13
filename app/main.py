from fastapi import FastAPI, UploadFile, File
from models.yolo.yolo_model import YOLOModel
import shutil
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# yolo 추론 받는 코드
@app.post("/predict/yolo")
async def predict_yolo(file: UploadFile = File(...)):
    # input / output 경로 지정
    input_path = f"temp/input_{file.filename}";
    output_path = f"temp/output_{file.filename}";
    
    # 디렉토리 없으면 생성
    os.makedirs("temp", exist_ok=True)

    # 파일 저장
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer);

    # 모델 호출
    model = YOLOModel();
    model.predict(input_path, output_path);