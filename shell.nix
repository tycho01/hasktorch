{ cudaSupport ? false, mklSupport ? true }:

let
  release = import ./build.nix { inherit cudaSupport mklSupport; };
in

with release;

pkgs.mkShell {
  name = "hasktorch-dev-environment";
  PATH               = "${cudatoolkit}/bin";
  LD_LIBRARY_PATH    = "${hasktorch-aten}/lib:${cudatoolkit}/lib64:${nvidia_x11-pinned}/lib";
  C_INCLUDE_PATH     = "${hasktorch-aten}/include:${cudatoolkit}/include";
  CPLUS_INCLUDE_PATH = "${hasktorch-aten}/include:${cudatoolkit}/include";
  buildInputs = [
    # pkgs.busybox
    pkgs.cabal-install
    (pkgs.haskellPackages.ghcWithPackages (self: with self; [ hspec-discover ]))

    hasktorch-aten
    hasktorch-codegen
    hasktorch-types-th
    hasktorch-ffi-th
    # these seem to have a runtime dependency error
    ] ; # ++ (lib.optionals cudaSupport [ hasktorch-types-thc hasktorch-ffi-thc ]);
}

