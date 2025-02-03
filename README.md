# sku110k-yolo11-to-onnx
Instructions to take the sku110k dataset alongside yolo11 and deploy it with onnx
nix-locate

# Container for training
A container is created because of mismatch cuda versions with pytorch

# Sources
https://docs.ultralytics.com/datasets/detect/sku-110k/
https://github.com/clementpoiret/nix-python-devenv/blob/cuda/devenv.nix
https://github.com/clementpoiret/nix-python-devenv/blob/cuda/devenv.yaml
https://devenv.sh/reference/yaml-options/
https://discourse.nixos.org/t/where-can-i-get-libgthread-2-0-so-0/16937
https://github.com/K4HVH/YOLOv11-ONNXRuntime-CPP