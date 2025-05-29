import shutil
import json
import time
from models.aqua_model import detect_aqua
from models.yolo_model import YOLOModel
from schemas.schemas import Polygon

async def handle_mix_detection(file, polygon_json):
    filename = f"{int(time.time() * 1000)}_{file.filename}"
    
    input_path = f"temp/Mix/input/{file.filename}"
    output_path = f"temp/Mix/output/{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    polygon_data = [Polygon(**item) for item in json.loads(polygon_json)] if polygon_json else []

    model = YOLOModel()
    data = model.predict(input_path, output_path, polygon_data);

    final_output_path = f"temp/Mix/output/{filename}"
    detect_aqua(output_path, final_output_path, polygon_data);

    print(output_path)

    return {"output": output_path, "data": data}
