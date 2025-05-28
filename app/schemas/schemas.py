from pydantic import BaseModel
from typing import List

class Point(BaseModel) :
    x: float
    y: float

class Polygon(BaseModel):
    id: int
    name: str
    points: List[Point]