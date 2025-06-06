# YOLO11-to-ONNX

This repository will be used as a guide to convert YOLO11 to ONNX using ultralitics to train a model and then use C++ with ONNX for inference, some details that I notice will be recorded here for future guidance.

This repo is meant for myself, but I will try to make it so anyone can follow the process easily, I will add the resources I used to guide me.

# NIX

I am using NIXOS, this distro has the nix package manager, this will be used to make the compilation of C++ easier, since the packages required can be added to a `nix.shell` file and will be found at compile time.

The program itself can also be made available as a package, this means that it will be compiled and available as a binary in the path when you enter the shell created by nix.

The use of `nix-shell` can make specific CUDA versions available directly, this is important for some libraries that will only work with older CUDA versions.

# EXTRA

The extra info is in a README file for training and another for the C++ implementation (one README per folder)

# KEY LEARNINGS AND SKILLS SHOWN IN THIS REPO

- The use of `nix-shell` with a specific cuda version.
- Training with `ultralytics` and exporting the model to `onnx`.
- The use of `onnxruntime` alongside C++.
- The use of `cmake` and `make` to compile C++ code with `nix` to create a package.
- The use of `nix` to create a container, although simple this shows the potential to use it for deployment on web environments.
- How to create a package using `nix` and how to use it in a shell or containerized environment.
- The use of `uv` to manage python packages.
- Some specific details about YOLO11, mainly the format of the image used by it when used with ONNX.
- The usage of CUDA tools within `nix`, in particular, the use of older versions os cuda within a system that has newer versions installed, reducing the overhead that other methods would have (like using containers) without impacting the parent system (like what installing multiple version in the main system would do).

# NOTE

About the usage of `nix`, tis was developed using the `https://nixos.org/channels/nixos-unstable` channel, at the time of development it was the beta for the `25.05` release. This is in case of incompatibility with future releases, this can be solved using `flakes`, that provide the most stable way to reproduce a development environment as it pins the versions of everything, this is not done by `nix-shell` alone, `nix-shell` pins the versions to when the environment was created localy, so there is improvement to be made in that area, this will be part of my next learning project.

# Sources

- https://docs.ultralytics.com/datasets/detect/sku-110k/
- https://github.com/clementpoiret/nix-python-devenv/blob/cuda/devenv.nix
- https://github.com/clementpoiret/nix-python-devenv/blob/cuda/devenv.yaml
- https://devenv.sh/reference/yaml-options/
- https://discourse.nixos.org/t/where-can-i-get-libgthread-2-0-so-0/16937
- https://github.com/K4HVH/YOLOv11-ONNXRuntime-CPP

Important for cuda to have a fixed version, the version numbers work despite them not being in the nix package search page
- https://nixos.org/manual/nixpkgs/unstable/#cuda