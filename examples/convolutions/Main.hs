{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module Main where

import Numeric.Backprop
import Data.Singletons.Prelude.Num
import Control.Monad.Trans
import Control.Monad.Trans.Maybe

import Torch.Cuda.Double as Torch

type Batch = 2
type KW = 2
type DW = 2
type Input = 5
type Output = 2
type Sequence1 = 7
type Sequence2 = 13

params :: Proxy '[Input, Output, KW, DW]
params = Proxy 

main :: IO ()
main = do
  directFunctionCalls
  usingBackpack

usingBackpack :: IO ()
usingBackpack = do
  conv <- initConv1d params

  -- do a forward pass
  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT getCosVec

  -- backprop manages our forward and backward passes
  let (o1, g1) = backprop (conv1d conv) input
  shape o1 >>= print
  shape g1 >>= print

  -- do a backprop pass with batch data
  Just (binput :: Tensor '[Batch, Sequence1, Input]) <- runMaybeT getCosVec
  let (o2, g2) = backprop (conv1dBatch conv) binput
  shape o2 >>= print
  shape g2 >>= print


-- ========================================================================= --
-- Example directly using functions

directFunctionCalls :: IO ()
directFunctionCalls = do
  conv <- initConv1d params

  -- do a forward pass
  Just (input :: Tensor '[Sequence2, Input]) <- runMaybeT getCosVec
  o1 <- conv1d_forward input conv
  shape o1 >>= print

  -- initialize a gradient input
  gw :: Tensor '[Sequence2, Output] <- constant 1
  shape gw >>= print

  -- do a backward pass
  o1' :: Tensor '[13, 5] <- conv1d_backward input gw conv
  shape o1' >>= print

  -- do a forward pass with batch data
  Just (binput :: Tensor '[Batch,Sequence1, Input]) <- runMaybeT getCosVec
  o2 <- conv1d_forwardBatch binput conv
  shape o2 >>= print

  -- initialize a gradient input
  gw :: Tensor '[Batch, Sequence1, Output] <- constant 1
  shape gw >>= print

  -- do a backwards pass with batch data
  o2' :: Tensor '[2,7,5] <- conv1d_backwardBatch binput gw conv
  shape o2' >>= print

-- ========================================================================= --
-- utility functions for this exercise

-- Make a rank-1 tensor containing the cosine reshaped to the desired dimensionality.
getCosVec :: forall d . Dimensions d => KnownNatDim (Product d) => MaybeT IO (Tensor d)
getCosVec = do
  t <- MaybeT . pure . fromList $ [1..(fromIntegral $ natVal (Proxy :: Proxy (Product d)))]
  lift $ Torch.cos t

-- simple initilizer
initConv1d
  :: KnownNatDim3 s o (s * kW)
  => Proxy '[s,o,kW,dW]
  -> IO (Conv1d s o kW dW)
initConv1d _ =
  fmap Conv1d $ (,)
    <$> Torch.uniform (-10::Double) 10
    <*> Torch.constant 1


