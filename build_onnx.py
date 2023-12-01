from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.fuse()  
model.info(verbose=True)  # Print model information
model.export(format='onnx')  # Export to ONNX
