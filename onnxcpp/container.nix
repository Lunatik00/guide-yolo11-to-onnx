{
  pkgs ? import <nixpkgs> {},
}:
pkgs.dockerTools.streamNixShellImage {
  name = "onnx-cpp";
  tag = "latest";
  drv = pkgs.mkShell {
  packages = [
    (pkgs.callPackage ./package.nix {})
  ] ;
};
}