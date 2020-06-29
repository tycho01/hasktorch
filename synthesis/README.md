# Typed Neuro-Symbolic Program Synthesis for the Typed Lambda Calculus

[![Build Status](https://travis-ci.com/tycho01/synthesis.svg?branch=master)](https://travis-ci.com/tycho01/synthesis)

## Links

- introductory [slides](https://docs.google.com/presentation/d/1gS3sDgF7HPkiTnE9piQ6IDSFm6idGD7MaXalYzw9BC0/edit?usp=sharing)
- the accompanying [paper](https://github.com/tycho01/thesis), my AI master thesis at UvA
- the [Neuro-Symbolic Program Synthesis](https://arxiv.org/abs/1611.01855) paper I'm building on top of
- [spreadsheet](https://docs.google.com/spreadsheets/d/1uDA9suwASDzllxJZDt--wZ0ci7q4eJIfPcAw9qr18-U/edit?usp=sharing) used a bit in design
- Haddock [documentation](https://tycho01.github.io/synthesis/) for the codebase
- [roadmap](https://github.com/tycho01/synthesis/projects/1)

## Usage

You can build and run this project using [Nix](https://nixos.org/nix/) + [Cabal](https://www.haskell.org/cabal/).

``` sh
# cabal
ghcup install latest
ghcup set latest
./setup-cabal.sh
cabal v1-update

# stack
# workaround for https://github.com/commercialhaskell/stack/issues/5134
stack build --test --file-watch 2>&1 | sed '/^Warning:/,/Invalid magic: e49ceb0f$/d'

# install Nix, Cachix:
bash <(curl https://nixos.org/nix/install)
. ~/.nix-profile/etc/profile.d/nix.sh
nix-env -iA cachix -f https://cachix.org/api/v1/install
# nixGL for GPU thru Nix: https://github.com/guibou/nixGL

# increase the solver limit in Cabal
vim ~/.cabal/config
# under `program-default-options` uncomment `ghc-options` and add: `-fconstraint-solver-iterations=8`

# enter dev shell
cachix use tycho01
nix-build # | cachix push tycho01
# cpu:
nix-shell
# gpu:
nixGLNvidia nix-shell --arg cudaVersion 10
# remake Cabal file any time you add/move files in package.yml:
hpack --force

# basic commands
stack build
stack test
stack repl lib:synthesis
stack run generator   -- --help
stack run synthesizer -- --help

# Generate documentation.
stack build --enable-documentation

# Profile
stack build --enable-profiling --ghc-options="-fno-prof-auto"
`stack exec which generator` +RTS -p

# viz profiling
stack install ghc-prof-flamegraph
ghc-prof-flamegraph generator.prof
stack install profiterole
profiterole generator.prof

# plot synthesis results
cd plotting/
pip install -r requirements.txt
cd ../
python plotting/plot.py
# makes plots for ./results/*.csv
```
