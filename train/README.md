# OVERVIEW

In this project the dataset `medical-pills` is used to demonstrate the training process, the trained model is exported to ONNX and then it is used with C++ to process an image and detect pills in it.

# TOOLS AND LIBRARIES

This section of the repo uses `uv` to manage Python libraries and versions, this makes it easy to manage.

The reason why `nix-shell` is used for training is to manage the correct cuda version directly, this is due to pytorch requirements.

The most important requirements to know and understand are:

- `ultralitics`: A library that has several models integrated with tools to export directly to ONNX, while I was working in this the library was updated and now it is easier to export the model while keeping non-maximum suppression, this saves time when using it, but it would still be recomended to add it manually to manage some values if your dataset need to detect more than 300 objects in one image, or to explore the generated model to know how to change the values for the nms function inside the onnx network.
- `pytorch`: The library that is used by ultralitics to train YOLO, it is important to know how to install the correct version to take advantage of your GPU, this can be a CUDA or ROCm GPU. I haven't tested ultralitics with the ROCm precompiled binaries of pytorch, but it should work.
- `onnx`: The library that is used to export the model to ONNX, this library was made with the intent to create a standarized format for machine learning models, and that is why it is available in different programming languages.
- `onnxslim`: an extra library, it can be added to optimize the model during export.
- `onnxruntime`: The library that is used to run the model after it has been esported to onnx.

In general, the GPU compatible binaries are also compatible with CPU, that is why I only used those when added as dependencies to the project, but it is recomended to use the CPU binaries if you will not be using GPU because of the differnce in size.

# STEP BY STEP

The first step is to enter the shell, using the command `nix-shell` within the training folder will start a shell containing the correct cuda versions for this project and the `uv` package to manage it you might need to allow unfree packages if you haven't done it yet, just follow the instructions in the error message if that happens.

To run a traing you just need to type `uv run train_yolo11.py`, this Python script will download the pretrained model `yolo11n.pt` and start the trainig process, this training process will look at the file `medical-pills.yaml`, this file defines the dataset and it has the link to download it, which will be done before training starts. This and other YAML files can be found in the ultralitics page, I suggest to read them to fully understand how to adapt a custom dataset to be used here, specially the SKU110K dataset because it has a custom script about what to do with the data after download to comply with the format requirements for this training process.

Due to the hard coded values the training will create a folder called `medical_pills/yolo11n` that has the data from the trainig, there is a folder called `weights` and the model we will use is called `best.pt`.

To export the model to ONNX you will use the script `export.py`, this script will take a `.pt` model and convert it to `.onnx`, some details are hardcoded, it is set to have an image size of 640 and to be exported with nms, since this is just a guide I will forego the optimizations to the script for general use, to cover that there is a CLI created by ultralitics (https://docs.ultralytics.com/usage/cli/#export).

To run the export you will use `uv run export.py medical_pills/yolo11n/weights/best.pt`, this script will take the model and export it to ONNX, the output is a file called `best.onnx` and will be found alongside the input file.

IMPORTANT: There is an error because the dependency `onnxruntime` is covered with `onnxruntime-gpu` and there is no `onnxruntime` installed, this is a bug on the `ultralitics` library, the model does export without problems.

# AFTER EXPORTING THE MODEL

We are done with the part of the project that uses Python, now we will start the C++ part of the project, to make it easier I recomend to make a copy of the model to have it in the same folder as the C++ code for easy access, but it is not required. You can use the `inference.py` script if you want to visualize the results of the model, it is a simple script that uses the ultralitics library alongside onnx to load the model and process an image, the results are saved in the `medical_pills` folder.

This project continues in the folder `onnxcpp`, where C++ is used to make inferences using ONNX.

# Sources

- https://docs.ultralytics.com/datasets/detect/medical-pills/
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/export/
- https://docs.ultralytics.com/usage/cli