#848x480
import json
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import ParkingHandle

app = FastAPI()

parkingmanager = ParkingHandle.ParkingManagement(
    model="models/yolo12m.pt",
      classes=[2]
)

@app.get("/")
def root():
    return {"message": "Server is running!"}


@app.get("/cam1")
def root():
    with open("status/cam1.json", "r") as f:
        data = json.load(f)
        return data


@app.get("/cam2")
def root():
    with open("status/cam2.json", "r") as f:
        data = json.load(f)
        return data


@app.post("/cam1")
async def process_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = parkingmanager(im0, "boxes/cam1.json", 1)
    print(results)
    return {
        "filled_slots": results.filled_slots if hasattr(results, "filled_slots") else None,
        "available_slots": results.available_slots if hasattr(results, "available_slots") else None,
    }

@app.post("/cam2")
async def process_frame(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = parkingmanager(im0, "boxes/cam1.json", 2)
    return {
        "filled_slots": results.filled_slots if hasattr(results, "filled_slots") else None,
        "available_slots": results.available_slots if hasattr(results, "available_slots") else None,
    }

# uvicorn server:app --host 0.0.0.0 --port 8000
