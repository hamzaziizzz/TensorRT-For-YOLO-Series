# Optimizing YOLOv8 with TensorRT

## Clone the repository

```git
git clone https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git YOLOv8-TensorRT
```

## Create requirements.txt

`Copy + Paste` the following code into a new file called `requirements.txt`:

```txt
fastapi
uvicorn
pycuda
cuda-python
numpy~=1.22.4
opencv-python~=4.5.5.64
Pillow~=10.1.0
matplotlib~=3.7.4
```

## Create Dockerfile

`Copy + Paste` the following code into a new file called `Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1

# Step 1 - Select base docker image
# FROM python:3.8-slim-buster
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Step 2 - Upgrade pip
RUN python3 -m pip install --upgrade pip --no-cache-dir
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade wheel
RUN pip3 install ultralytics
RUN pip3 install install --upgrade nvidia-tensorrt
RUN pip3 uninstall opencv -y

# Step 3 - Set timezone, for updating system in subsequent steps
#https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Step 4 - Change work directory in docker image
WORKDIR /YOLOv8-TensorRT

# Step 5 - Upgrade and install additional libraries to work on videos
RUN apt update -y && apt upgrade -y
RUN apt install ffmpeg libsm6 libxext6 cmake build-essential -y

# Step 6 - Copy and install project requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Step 7 - Uncomment following line if we want to copy project contents to docker image
#COPY . .

# Step 8 - Give starting point of project
#CMD ["python3", "src/main.py"]
# For just starting the container (i.e., we will have to manually execute the program from inside the container), uncomment next line
CMD ["sh"]
```

## Build Docker Image

```docker
# Already existing image name is "yolov8-tensorrt"
docker build -t <image-name> .
```

For example,

```docker
docker build -t yolov8-tensorrt .
```

## Run Docker Image

```docker
# Already existing container name is "yolov8-tensorrt-container"
docker run -p <port>:6969 --gpus device=<gpu_id> --init -it --entrypoint sh -v $(pwd):/YOLOv8-TensorRT --rm --name <container_name> <image_name>
```

For example,

```docker
docker run -p 6969:6969 --gpus device=0 --init -it --entrypoint sh -v $(pwd):/YOLOv8-TensorRT --rm --name yolov8-tensorrt-container yolov8-tensorrt
```

## Export ONNX

Copy + Paste the following code into a new file called `build_onnx.py`:

```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.fuse()  
model.info(verbose=True)  # Print model information
model.export(format='onnx')  # Export to ONNX
```

Run the following command:

```bash
python3 build_onnx.py
```

## Generate TRT File

Run the following command:

```bash
python export.py -o yolov8x.onnx -e yolov8x.trt --end2end --v8
```

### Inference

```bash
# For image
python trt.py -e yolov8x.trt  -i src/1.jpg -o yolov8n-1.jpg --end2end

# For video
python trt.py -e yolov8x.trt  -i src/video1.mp4 -o reaults.avi --end2end
```

***Note: If getting `pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?`, then add the following lines to `utils/utils.py` after **line 32**:***

```python
cuda.init()
cuda.Device(0).make_context()
```

## Batch Inference

Add the following code to `utils/utils.py` after **line 102** (*i.e., after `detect_video()` function*):

```python
def batch_mobile_detection(self, batch_of_frames, conf=0.5, end2end=True):
        batch_results = []
        for frame in batch_of_frames:
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            data = self.infer(blob)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5 + self.n_classes)))[0]
                dets = self.postprocess(predictions, ratio)

            mobile_phone_locations = []
            if dets is not None:
                for det in dets:
                    x1, y1, x2, y2 = det[:4]
                    confidence = det[4]
                    cls_id = int(det[5])
                    if cls_id == 67 and confidence >= conf:
                        mobile_phone_locations.append([x1, y1, x2, y2])

            batch_results.append(mobile_phone_locations)

        return batch_results
```

## Build API

Copy + Paste the following code into a new file called `api.py`:

```python
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
```

## Run API

```bash
python3 api.py
```

## Test API

Open a new terminal and `copy + paste` the following code into a new file called `test_api.py`:

```python
import cv2
import requests
import numpy as np
import base64

# URL of the FastAPI endpoint
api_url = "http://192.168.12.1:6969/detect_batch/"

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.18.93:554/stream1")

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    batch_size = 32
    frames_batch = []

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from the webcam.")
            break

        # encode each frame in JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        # get base64 formatted data
        data = base64.b64encode(buffer).decode("ascii")
        frames_batch.append(data)

        # If the batch size is reached, send the batch to the FastAPI endpoint
        if len(frames_batch) == batch_size:
            # Prepare the JSON payload
            payload = {"frames": frames_batch}

            # Send a POST request to the FastAPI endpoint
            response = requests.post(api_url, json=payload)

            # Check the response
            if response.status_code == 200:
                result = response.json()
                print(result)
            else:
                print(f"Error: {response.status_code} \n {response.text}")

            # Reset the frames batch
            frames_batch = []

        # Display the frame
        # cv2.imshow("Webcam", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
```
