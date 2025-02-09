# simple.nix
{pkgs}:

pkgs.clangStdenv.mkDerivation {
name = "onnx_c++";

  src = ./.;

  buildInputs = with pkgs; [
    cmake
    libGL
    glib
    libuv
    zlib
    onnxruntime
    (opencv.override{enableGtk3=true;}) # Opencv with gtk3 enables window creation
    argparse
  ] ;
  nativeBuildInputs = with pkgs; [
    libGL
    glib
    libuv
    zlib
    onnxruntime
    (opencv.override{enableGtk3=true;}) # Opencv with gtk3 enables window creation
  ] ;
  # The deletion of the build folder is due to a possible conflict if it exists, 
  # it is added here in case you aren't using nix-shell at the begining but try to use it later
  buildPhase = ''
    rm -rf build
    cmake .
    make
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp onnx_c++   $out/bin/onnx_c++
  '';
}
