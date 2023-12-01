from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from utils.utils import BaseEngine
import numpy as np
import base64
import cv2

app = FastAPI()

# Load the YOLOv8 Mobile Detection Model
engine_path = "yolov8x.trt"  # Update with the actual path
yolo_engine = BaseEngine(engine_path)


class BatchRequest(BaseModel):
    frames: List[str]  # List of base64-encoded JPEG images


@app.post("/detect_batch/")
async def detect_batch(request: BatchRequest):
    try:
        # Decode base64 and convert to NumPy array
        batch_of_frames = [cv2.imdecode(np.frombuffer(base64.b64decode(frame), dtype=np.uint8), cv2.IMREAD_COLOR) for frame in request.frames]

        # Perform batch detection
        batch_results = yolo_engine.batch_mobile_detection(batch_of_frames)

        return JSONResponse(content=batch_results, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6969)
