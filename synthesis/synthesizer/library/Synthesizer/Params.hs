{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}

module Synthesizer.Params (module Synthesizer.Params) where

import GHC.TypeNats (type (+))
import Torch.Typed.Aux
import Synthesis.Data

-- TODO: consider which hyperparams have been / should be shared across networks

-- | left-hand symbols: just Expression in our lambda-calculus DSL
type LhsSymbols = 1

-- | must use a static batch size i/o making it dynamic by SynthesizerConfig...
type EncoderBatch = 8
encoderBatch :: Int
encoderBatch = natValI @EncoderBatch
-- type R3nnBatch = 8  -- moved to generator
r3nnBatch :: Int
r3nnBatch = natValI @R3nnBatch

-- R3NN
-- static LSTM can't deal with dynamic number of layers, as it unrolls on compile (init/use)
type NumLayers = 3