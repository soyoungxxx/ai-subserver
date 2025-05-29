import shutil
import time;
from models.yolo_model import YOLOModel

async def handle_yolo_detection(file):
    filename = f"{int(time.time() * 1000)}_{file.filename}"
    
    input_path = f"temp/Yolo/input/{file.filename}"
    output_path = f"temp/Yolo/output/{filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model = YOLOModel()
    model.predict(input_path, output_path, polygon_data=None)

    return {"output" : output_path, "data" : None}
