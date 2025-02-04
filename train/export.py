from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Simple scrip to export model to onnx")
parser.add_argument("filename")
args = parser.parse_args()
# Load the YOLO11 model
model = YOLO(args.filename)

# Export the model to ONNX format
model.export(format="onnx", nms=True, imgsz=640)  # creates 'yolo11n.onnx'
