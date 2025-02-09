import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Simple scrip to make a single inference using ultralitics")
parser.add_argument("model_file", type=str, help="Model file path")
parser.add_argument("image_file", type=str, help="Image file path")
args = parser.parse_args()
# Load the exported ONNX model
model = YOLO(args.model_file)
# Run inference
pred = model.predict(   args.image_file,         
                        project="medical_pills",
                        name="inference",
                        save=True)