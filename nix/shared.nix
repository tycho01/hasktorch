{ compiler ? "ghc865" }:

let
  libtorch_src = pkgs:
    let src = pkgs.fetchFromGitHub {
          owner  = "stites";
          repo   = "pytorch-world";
          rev    = "ebfe09208964af96ae4d5cf1f70d154b16826c6e";
          sha256 = "1x3jn55ygggg92kbrqvl9q8wgcld8bwxm12j2i5j1cyyhhr1p852";
    };
    in (pkgs.callPackage "${src}/libtorch/release.nix" { });

  overlayShared = pkgsNew: pkgsOld: {
    inherit (libtorch_src pkgsOld)
      libtorch_cpu
      libtorch_cudatoolkit_9_2
      libtorch_cudatoolkit_10_1
    ;

    haskell = pkgsOld.haskell // {
      packages = pkgsOld.haskell.packages // {
        "${compiler}" = pkgsOld.haskell.packages."${compiler}".override (old: {
            overrides =
              let
                appendConfigureFlag = pkgsNew.haskell.lib.appendConfigureFlag;
                dontCheck = pkgsNew.haskell.lib.dontCheck;
                failOnAllWarnings = pkgsNew.haskell.lib.failOnAllWarnings;
                overrideCabal = pkgsNew.haskell.lib.overrideCabal;
                optionalString = pkgsNew.stdenv.lib.optionalString;
                isDarwin = pkgsNew.stdenv.isDarwin;

                mkHasktorchExtension = postfix:
                  haskellPackagesNew: haskellPackagesOld: {
                    "libtorch-ffi_${postfix}" =
                        appendConfigureFlag
                          (overrideCabal
                            (haskellPackagesOld.callCabal2nix
                              "libtorch-ffi"
                              ../libtorch-ffi
                              { c10 = pkgsNew."libtorch_${postfix}"
                              ; torch = pkgsNew."libtorch_${postfix}"
                              ; }
                            )
                            (old: {
                                preConfigure = (old.preConfigure or "") + optionalString isDarwin ''
                                  sed -i -e 's/-optc-std=c++11 -optc-xc++/-optc-xc++/g' ../libtorch-ffi/libtorch-ffi.cabal;
                                '';
                              }
                            )
                          )
                        "--extra-include-dirs=${pkgsNew."libtorch_${postfix}"}/include/torch/csrc/api/include";
                    "hasktorch_${postfix}" =
                      overrideCabal
                        (haskellPackagesOld.callCabal2nix
                          "hasktorch"
                          ../hasktorch
                          { libtorch-ffi = haskellPackagesNew."libtorch-ffi_${postfix}"; }
                        )
                        (old: {
                              preConfigure = (old.preConfigure or "") + optionalString (!isDarwin) ''
                                export LD_PRELOAD=${pkgs.mkl}/lib/libmkl_rt.so
                              '';
                            }
                        );
                    "hasktorch-examples_${postfix}" =
                      # failOnAllWarnings
                        (haskellPackagesOld.callCabal2nix
                          "examples"
                          ../examples
                          { libtorch-ffi = haskellPackagesNew."libtorch-ffi_${postfix}"
                          ; hasktorch = haskellPackagesNew."hasktorch_${postfix}"
                          ; }
                        );
                  };

                extension =
                  haskellPackagesNew: haskellPackagesOld: {
                    hasktorch-codegen =
                      # failOnAllWarnings
                        (haskellPackagesNew.callCabal2nix
                          "codegen"
                          ../codegen
                          { }
                        );
                    inline-c =
                      # failOnAllWarnings
                        (haskellPackagesNew.callHackageDirect
                          {
                            pkg = "inline-c";
                            ver = "0.9.0.0";
                            sha256 = "07i75g55ffggj9n7f5y6cqb0n17da53f1v03m9by7s4fnipxny5m";
                          }
                          { }
                        );
                    inline-c-cpp =
                      # failOnAllWarnings
                      dontCheck
                        (overrideCabal
                          (haskellPackagesNew.callHackageDirect
                            {
                              pkg = "inline-c-cpp";
                              ver = "0.4.0.0";
                              sha256 = "15als1sfyp5xwf5wqzjsac3sswd20r2mlizdyc59jvnc662dcr57";
                            }
                            { }
                          )
                          (old: {
                              preConfigure = (old.preConfigure or "") + optionalString isDarwin ''
                                sed -i -e 's/-optc-std=c++11//g' inline-c-cpp.cabal;
                              '';
                            }
                          )
                        );
                  };

              in
                pkgsNew.lib.fold
                  pkgsNew.lib.composeExtensions
                  (old.overrides or (_: _: {}))
                  [ (pkgsNew.haskell.lib.packagesFromDirectory { directory = ./haskellExtensions/.; })
                    extension
                    (mkHasktorchExtension "cpu")
                    (mkHasktorchExtension "cudatoolkit_9_2")
                    (mkHasktorchExtension "cudatoolkit_10_1")
                  ];
          }
        );
      };
    };
  };

  bootstrap = import <nixpkgs> { };

  nixpkgs = builtins.fromJSON (builtins.readFile ./nixpkgs.json);

  src = bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo  = "nixpkgs";
    inherit (nixpkgs) rev sha256;
  };

  pkgs = import src {
    config = { allowUnsupportedSystem = true; allowUnfree = true; };
    overlays = [ overlayShared ];
  };

  nullIfDarwin = arg: if pkgs.stdenv.hostPlatform.system == "x86_64-darwin" then null else arg;

  fixmkl = old: old // {
      shellHook = ''
        export LD_PRELOAD=${pkgs.mkl}/lib/libmkl_rt.so
      '';
    };
  fixcpath = libtorch: old: old // {
      shellHook = ''
        export CPATH=${libtorch}/include/torch/csrc/api/include
      '';
    };
in
  rec {
    inherit nullIfDarwin;

    inherit (pkgs.haskell.packages."${compiler}")
      hasktorch-codegen
      inline-c
      inline-c-cpp
      libtorch-ffi_cpu
      libtorch-ffi_cudatoolkit_9_2
      libtorch-ffi_cudatoolkit_10_1
      hasktorch_cpu
      hasktorch_cudatoolkit_9_2
      hasktorch_cudatoolkit_10_1
      hasktorch-examples_cpu
      hasktorch-examples_cudatoolkit_9_2
      hasktorch-examples_cudatoolkit_10_1
    ;
    hasktorch-docs = ((import ./haddock-combine.nix {runCommand = pkgs.runCommand; lib = pkgs.lib; haskellPackages = pkgs.haskellPackages;}) {hspkgs = [pkgs.haskell.packages."${compiler}".hasktorch_cpu];});
    shell-hasktorch-codegen = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-codegen).env;
    shell-inline-c = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c).env;
    shell-inline-c-cpp = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".inline-c-cpp).env;
    shell-libtorch-ffi_cpu = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi_cpu).env.overrideAttrs (fixcpath pkgs.libtorch_cpu);
    shell-libtorch-ffi_cudatoolkit_9_2 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi_cudatoolkit_9_2).env.overrideAttrs (fixcpath pkgs.libtorch_cudatoolkit_9_2);
    shell-libtorch-ffi_cudatoolkit_10_1 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".libtorch-ffi_cudatoolkit_10_1).env.overrideAttrs (fixcpath pkgs.libtorch_cudatoolkit_10_1);
    shell-hasktorch_cpu = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch_cpu).env.overrideAttrs fixmkl;
    shell-hasktorch_cudatoolkit_9_2 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch_cudatoolkit_9_2).env.overrideAttrs fixmkl;
    shell-hasktorch_cudatoolkit_10_1 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch_cudatoolkit_10_1).env.overrideAttrs fixmkl;
    shell-hasktorch-examples_cpu = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-examples_cpu).env.overrideAttrs fixmkl;
    shell-hasktorch-examples_cudatoolkit_9_2 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-examples_cudatoolkit_9_2).env.overrideAttrs fixmkl;
    shell-hasktorch-examples_cudatoolkit_10_1 = (pkgs.haskell.lib.doBenchmark pkgs.haskell.packages."${compiler}".hasktorch-examples_cudatoolkit_10_1).env.overrideAttrs fixmkl;
  }

