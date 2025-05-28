import shutil
from models.aqua_model import detect_aqua
from models.yolo_model import YOLOModel
from schemas.schemas import Polygon

async def handle_mix_detection(file, polygon_json):
    input_path = f"temp/Mix/input/{file.filename}"
    output_path = f"temp/Mix/output/{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    polygon_data = Polygon.parse_raw(polygon_json) if polygon_json else None

    # 수체 - 객체 순서대로
    # detect_aqua(input_path, output_path);
    print("im here")

    model = YOLOModel()
    model.predict(output_path, output_path, polygon_data=None)

    return {"output": output_path, "data": polygon_data}
