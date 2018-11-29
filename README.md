# hasktorch

[![Stable Status](https://circleci.com/gh/hasktorch/hasktorch/tree/master.svg?style=shield&circle-token=9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/hasktorch/hasktorch/tree/master)
[![Development Status](https://circleci.com/gh/hasktorch/hasktorch/tree/dev.svg?style=shield&circle-token=9455d7cc953a58204f4d8dd683e9fa03fd5b2744)](https://circleci.com/gh/hasktorch/hasktorch/tree/dev)

Hasktorch is a library for tensors and neural networks in Haskell. It is an independent open source community project which leverages the core C libraries shared by [Torch](http://torch.ch/) and [PyTorch](http://pytorch.org/). This library leverages cabal new-build and [backpack][backpack].

[backpack]: https://github.com/ezyang/ghc-proposals/blob/backpack/proposals/0000-backpack.rst

**Note: This project is in early development and should only be used by contributing developers. Expect substantial changes to the library API as it evolves. Contributions and PRs are welcome (see details below).**

## Project Organization

In order of high-to-low level. The reverse order can also be seen in the `cabal.project` file.

| Directory                   | Description |
| --------------------------- | ----------- |
| [`examples/`][examples]     | Examples of basic usage and experimental prototypes |
| [`hasktorch/`][hasktorch]   | Reexports of the high-level interface to basic tensor creation and math operations and manages allocation/deallocation via foreign pointers |
| [`indef/`][indef]           | Orphan instances of the above typeclasses for the relevant backpack signatures |
| [`signatures/`][signatures] | Backpack signatures which line up with the generated C-FFI |
| [`types/`][types]           | Memory-managed tensors and core data types that satisfy global and type-specific backpack types |
| [`zoo/`][zoo]               | Beginnings of a model zoo |

## Build Instructions

Currently hasktorch only supports OSX and Linux builds because these are what the development team works on -- if you would like to add *BSD or Windows support, please let us know!

Building Hasktorch requires retrieving and building a fork of pytorch's [ATen library][aten] dependency. Currently we fork ATen to reintroduce some C-native broadcasting functionality which was moved to ATen's C++ core. This step has been condensed in our Makefile as `make init`.

[aten]:https://github.com/zdevito/ATen/

Following this, you will need cabal-install > 2.0 for `new-build` and backpack support. We would like to support stack in the future, but that is pending  the completion of [backpack support in stack][stack-backpack].

[cabal-latest]:https://github.com/haskell/cabal/tree/30d0c10349b6cc69adebfe06c230c784574ebf7a
[stack-backpack]:https://github.com/commercialhaskell/stack/issues/2540

Now you can build hasktorch:

```
cabal new-build all
cabal new-run static-tensor-usage
```

To build without GPU support/CUDA, use:

```
cabal new-build all --flags=-cuda
cabal new-run static-tensor-usage --flags=-cuda
```

Due to the aten shared library and use of new cabal functionality with backpack, the setup process can be a bit more involved than a typical haskell library. Don't hesitate to reach out to the development team for questions or issues with getting setup (see Contributing below). 

## Getting started

For an example of basic end-user API usage, see the [statically typed
tensor usage][static-tensor-usage] example and the example of [simple linear regression using backprop][backprop-regression].

[static-tensor-usage]: https://github.com/hasktorch/hasktorch/blob/master/examples/static-tensor-usage/Main.hs
[backprop-regression]:https://github.com/hasktorch/hasktorch/blob/master/examples/backprop-regression/BackpropRegression.hs

For details on implementation and usage of raw C bindings,
refer to their respective README documentation in [the ffi reposotory][ffi].

[ffi]:https://github.com/hasktorch/ffi

Additional examples can be found in [`examples/`][examples] as well as the test modules.

## Contributing

We welcome new contributors. For a rough list of outstanding items on deck
(there are many places to contribute), refer to:

https://github.com/hasktorch/hasktorch/projects/1

Contact Austin Huang or Sam Stites for access to the [hasktorch slack channel][slack]. You can find our contact information by digging through cabal files or you can ping us on gitter in the [DataHaskell Lobby][gitter-dh].

[slack]:https://hasktorch.slack.com 
[gitter-dh]:https://gitter.im/dataHaskell/Lobby

## Thanks

Thanks to all hasktorch developers who have contributed to this community effort so far. This project is also indebted to prior work on typed functional
programming for deep learning by
Justin Le [![][gh-icon]][mstkg-gh][![][blog-icon]][mstkg],
Edward Yang [![][gh-icon]][ezyang-gh][![][blog-icon]][ezyang],
Huw Campbell [![][gh-icon]][huw-gh][![][blog-icon]][huw],
Kaixi Ruan [![][gh-icon]][krpopo-gh],
and Khanh Nguyen [![][gh-icon]][khanhptnk-gh],
as well as to the Torch and PyTorch dev teams.

[gh-icon]:https://png.icons8.com/small/1x/github.png
[blog-icon]:https://png.icons8.com/small/1x/blog.png
[mstkg]:https://blog.jle.im/
[mstkg-gh]:https://github.com/mstksg
[huw-gh]:https://github.com/HuwCampbell
[huw]:http://www.huwcampbell.com/
[krpopo-gh]:https://github.com/krpopo
[khanhptnk-gh]:https://github.com/khanhptnk
[ezyang-gh]:https://github.com/ezyang/
[ezyang]:http://ezyang.com/

<!-- project directory links -->
[developers]: ./DEVELOPERS.md
[makefile]: ./Makefile
[types]: ./types/
[signatures]: ./signatures/
[hasktorch]: ./hasktorch/
[examples]: ./examples/
[indef]: ./indef/
[zoo]: ./zoo/
