# OVERVIEW

This section of the project shows how to use YOLO11n onnx model with C++.

# CMAKE

The `CMakeLists.txt` file is used to define the requirements for building the project, in this case we need the `onnxruntime` library and the `opencv` library for runtime and additionally the `argparse` library for build time compilation to add command line arguments.

# CODE DETAILS

We define a function to rezise an image to the size that the model requires as an an input. There is another function that uses a defined `Struct` to organize the results.

The code loads the onnx model, then it gets the data required to create the input and output vectors, then it reads the image and it rezises it, since we used `opencv` to read the image it is read as `BGR` and as `UINT8`, the `YOLO11n` model was trained with `RGB` and the image should be `FLOAT`, between `0` and `1`, so, the image is adjusted to fit the model and then it is assigned to the input vector. Following this we use `onnxruntime` to process the image and get the result, then it is sorted with the `sort_onnx_nms_output` function, the final step in the code draws the bounding boxes and save the image for visualization purposes.

# COMMAND LINE ARGUMENTS

The command line arguments are used to define the path of the model and the image that will be processed by the model. The following table shows the available options:

| Argument | Description                                      |
| -------- | ------------------------------------------------ |
| model     | Path to the onnx model                           |
| image     | Path to the image that will be processed by the model |

# How to use

I defined the code to be built and be a package using `nix` this is defined in `package.nix`, and this package is used to create a shell defined in `nix.shell`, when `nix-shell` is executed the system will build the project and make the binary available, the binary is called `onnx_c++`, so, to run the code we need to execute:

```bash
$ nix-shell
[nix-shell] $ onnx_c++ <path_to_model> <path_to_image>
```

# Container

Alternatively we can create a container as defined in `container.nix`, the container was mostly used to test how the creation of containers using `nix` works, so it is not very useful for this project, but if you want to try it out just execute:

```bash
$ nix-build container.nix && ./result | podman load
```

This will create a container with all the dependencies needed to run the code.

# SOURCES

- https://github.com/K4HVH/YOLOv11-ONNXRuntime-CPP
- https://github.com/NixOS/nixpkgs/blob/226be8a519e18880e5ddf218368780748f6d6677/pkgs/build-support/docker/examples.nix#L735
- https://discourse.nixos.org/t/create-docker-container-from-nix-shell/17441/6

