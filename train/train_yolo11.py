from urllib.request import urlretrieve
from pathlib import Path

from ultralytics import YOLO


def get_YOLO11n() -> str:
    model_file = "yolo11n.pt"
    model_url = (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    )
    if not Path(model_file).exists():
        urlretrieve(model_url, model_file)
    return model_file


def main():
    model_file = get_YOLO11n()
    # Load a model
    model = YOLO(model_file)  # load a pretrained model (recommended for training)

    # yaml file
    # data_url = "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml"
    dataset_file = "medical-pills.yaml"

    # urlretrieve(data_url, data_file)

    # Train the model, more options in https://docs.ultralytics.com/usage/cfg/#train-settings
    results = model.train(
        data=dataset_file,
        epochs=100,
        imgsz=640,
        multi_scale=False,  # Set to true if higher or lower imgz would be required during inference, otherwise it is recomended to use the same imgz for inference
        project="medical_pills",
        name="yolo11n",
        profile=True,
        plots=True,
        exist_ok=True,  # This makes it so the training will be overriden if done more than once
    )


if __name__ == "__main__":
    main()
