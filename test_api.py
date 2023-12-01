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
