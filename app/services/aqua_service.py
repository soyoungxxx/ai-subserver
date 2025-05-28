import shutil
from models.aqua_model import detect_aqua

async def handle_aqua_detection(file):
    input_path = f"temp/Aqua/input/{file.filename}"
    output_path = f"temp/Aqua/output/{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detect_aqua(input_path, output_path);

    return {"output" : output_path, "data" : None}
