#!/usr/bin/env bash

# - Retrieves a prebuilt libtorch binary per https://pytorch.org/cppdocs/installing.html
# - Retrieves a release binary for mkl https://github.com/intel/mkl-dnn/releases
#   which is a runtime dependency that is not package w/ libtorch

set -eu

USE_BINARY_FOR_CI="$1"

case "$(uname)" in
  "Darwin")
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.0.0.zip
    unzip libtorch-macos-1.0.0.zip
    rm libtorch-macos-1.0.0.zip
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_mac_2019.0.1.20181227.tgz
    tar -xzf mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz
    rm -f mklml_mac_2019.0.1.20181227.tgz.1
    mv mklml_mac_2019.0.1.20181227 mklml
    ;;
  "Linux")
    if [ -z "$USE_BINARY_FOR_CI" ] ; then
      wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    else
      wget https://github.com/hasktorch/libtorch-binary-for-ci/releases/download/1.0.0.dev20190319/libtorch-shared-with-deps-latest.zip
    fi
    unzip libtorch-shared-with-deps-latest.zip
    rm libtorch-shared-with-deps-latest.zip
    wget https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20181227.tgz
    tar -xzf mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz
    rm -f mklml_lnx_2019.0.1.20181227.tgz.1
    mv mklml_lnx_2019.0.1.20181227 mklml
    ln -s libmklml_intel.so mklml/lib/libmklml.so
    ;;
esac


# Following codes are copied from pytorch/tools/run-clang-tidy-in-ci.sh.
# Generate ATen files.
pushd pytorch

if [[ ! -d build ]]; then
mkdir build
fi

python aten/src/ATen/gen.py \
  -s aten/src/ATen \
  -d build/aten/src/ATen \
  aten/src/ATen/Declarations.cwrap \
  aten/src/THNN/generic/THNN.h \
  aten/src/THCUNN/generic/THCUNN.h \
  aten/src/ATen/nn.yaml \
  aten/src/ATen/native/native_functions.yaml

sed -i -e "s/ name: n$/ name: 'n'/g" -e "s/ name: N$/ name: 'N'/g" build/aten/src/ATen/Declarations.yaml

popd
