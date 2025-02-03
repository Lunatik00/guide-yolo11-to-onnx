podman build . -t train-yolo11
podman run  --rm -it --device nvidia.com/gpu=all --shm-size=4gb -v $PWD:/code train-yolo11