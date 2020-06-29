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
cabal update
source setenv
cabal install

# basic commands
cabal build
cabal test
cabal repl lib:synthesis
cabal run generator   -- --help
cabal run synthesizer -- --help

# Generate documentation.
cabal build --enable-documentation

# Profile
cabal build --enable-profiling --ghc-options="-fno-prof-auto"
`cabal exec which generator` +RTS -p

# viz profiling
cabal install ghc-prof-flamegraph
ghc-prof-flamegraph generator.prof
cabal install profiterole
profiterole generator.prof

# plot synthesis results
cd plotting/
pip install -r requirements.txt
cd ../
python plotting/plot.py
# makes plots for ./results/*.csv
```
