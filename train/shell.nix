let
  pkgs = import <nixpkgs> {};
  cuda_pkgs = with pkgs; [
    cudaPackages_12_4.cuda_cudart
    cudaPackages_12_4.cudatoolkit
    cudaPackages_12_4.cudnn
    cudaPackages_12_4.cuda_nvcc
    cudaPackages_12_4.setupCudaHook
    libGL
    glib
    libuv
    zlib];
in pkgs.mkShell {
  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath cuda_pkgs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages_12_4.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages_12_4.cudatoolkit; 
  };
  packages = [
    pkgs.uv
  ] ++ cuda_pkgs;
}