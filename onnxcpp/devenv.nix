{pkgs, ...}:
let
  cuda_pkg = pkgs.callPackage { pkgs.cudaPackages = pkgs.cudaPackages_12_4; };
  buildInputs = with pkgs; [
    cudaPackages_12_4.cuda_cudart
    cudaPackages_12_4.cudatoolkit
    cudaPackages_12_4.cudnn
    cudaPackages_12_4.cuda_nvcc
    cudaPackages_12_4.setupCudaHook
    libGL
    glib
    libuv
    zlib
    clang_19
    onnxruntime
    argparse
    (opencv.override{enableGtk3=true;}) # Opencv with gtk3 enables window creation
  ] ;
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
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages_12_4.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages_12_4.cudatoolkit; 
  };

  languages.cplusplus.enable = true; # Add c++ support
}
