from fastapi import FastAPI
from routes import yolo_router
from routes import aqua_router
from routes import mix_router

# CORS 설정
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 * 허용, 운영 시에는 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# predict/yolo
app.include_router(yolo_router.router, prefix="/predict/object");

# predict/aqua
app.include_router(aqua_router.router, prefix="/predict/water-body");

# predict/mix
app.include_router(mix_router.router, prefix="/predict/mix");