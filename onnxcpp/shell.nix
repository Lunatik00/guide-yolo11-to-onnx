let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.callPackage ./package.nix {})
  ] ;
}
