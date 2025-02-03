{pkgs, ...}:
let
  buildInputs = with pkgs; [
    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    libGL
    glib
    libuv
    zlib
    clang_19
    onnxruntime
    cudaPackages.cuda_nvcc
    (opencv.override{enableGtk3=true;}) # Opencv with gtk3 enables window creation
  ];
in 
{
  packages = buildInputs;
  stdenv = pkgs.llvmPackages_19.stdenv; # use clang instead of cpp for compilation
  # Env variables for GPU support
  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages.cudatoolkit; 
  };

 languages.python = {
  enable = true;
  version = "3.11";
  uv.enable = true;
  }; # The python scripts will be managed with uv
  languages.cplusplus.enable = true; # Add c++ support
  # dotenv.enable = true;

}
