from fastapi import FastAPI, UploadFile, File
from models.yolo.yolo_model import YOLOModel

app = FastAPI()
model = YOLOModel("yolov8n.pt")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/yolo")
async def predict_yolo(file: UploadFile = File(...)):
    contents = await file.read()
    result_json = model.predict(contents)
    return {"result": result_json}