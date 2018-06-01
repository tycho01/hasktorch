{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Torch.Indef.Static.NN.Layers where

import Data.List
import Numeric.Backprop

import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Blas
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN as Dynamic

-- datatype representing a linear layer with bias. Represents
-- @y = Ax + b@.
newtype Linear i o
  = Linear { getTensors :: (Tensor '[i, o], Tensor '[o]) }

instance KnownDim2 i o => Show (Linear i o) where
  show c = intercalate ","
    [ "Linear ("
    ++ "input: "  ++ show (inputSize c)
    , " output: " ++ show (outputSize c)
    ++ ")"
    ]

instance (KnownDim2 i o) => Backprop (Linear i o) where
  zero = const . Linear $ (constant 0, constant 0)
  one  = const . Linear $ (constant 1, constant 1)
  add c0 c1 = Linear (weights c0 + weights c1, bias c0 + bias c1)

-- | the dense weight matrix of a linear layer
weights :: Linear i o -> Tensor '[i, o]
weights (Linear (w, _)) = w

-- | the bias vector of a linear layer
bias :: Linear i o -> Tensor '[o]
bias (Linear (_, b)) = b

-- | The input size of a linear layer
inputSize :: forall i o . KnownDim i => Linear i o -> Int
inputSize _ = fromIntegral (dimVal (dim :: Dim i))

-- | The output size of a linear layer
outputSize :: forall i o kW dW . KnownDim o => Linear i o -> Int
outputSize _ = fromIntegral (dimVal (dim :: Dim o))

-- ========================================================================= --

-- | Backprop linear function without batching
linear
  :: forall s i o
  .  Reifies s W
  => KnownDim2 i o
  => BVar s (Linear i o)
  -> BVar s (Tensor '[i])
  -> BVar s (Tensor '[o])
linear = liftOp2 $ op2 $ \l i -> (transpose2d (weights l) `mv` i + bias l, go l i)
  where
    go :: Linear i o -> Tensor '[i] -> Tensor '[o] -> (Linear i o, Tensor '[i])
    go (Linear (w, b)) i gout = (Linear (i `outer` b', b'), w `mv` b')
      where
        b' = gout - b

flattenBP
  :: (Reifies s W, KnownDim (Product d), Dimensions d)
  => BVar s (Tensor d) -> BVar s (Tensor '[Product d])
flattenBP = liftOp1 . op1 $ \t -> (flatten t, resizeAs)

-- mmultBP
--   :: forall a b c s
--   .  (KnownDim3 a b c, Reifies s W)
-- 
--   => BVar s (Tensor '[a, b])
--   -> BVar s (Tensor '[b, c])
-- 
--   -> BVar s (Tensor '[a, c])
-- mmultBP = liftOp2 . op2 $ \a b ->
--   (a !*! b, \gout -> (gout !*! transpose2d b, transpose2d a !*! gout))

-------------------------------------------------------------------------------

_sparseLinear_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_updateOutput t0 t1 t2 t3 = Dynamic._sparseLinear_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
_sparseLinear_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
_sparseLinear_accGradParameters t0 t1 t2 t3 t4 t5 = Dynamic._sparseLinear_accGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

_sparseLinear_zeroGradParameters :: Tensor d -> Tensor d -> Tensor d -> IO ()
_sparseLinear_zeroGradParameters t0 t1 t2 = Dynamic._sparseLinear_zeroGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2)
_sparseLinear_updateParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_sparseLinear_updateParameters t0 t1 t2 t3 t4 = Dynamic._sparseLinear_updateParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

_gatedLinear_updateOutput :: Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateOutput t0 t1 = Dynamic._gatedLinear_updateOutput (asDynamic t0) (asDynamic t1)
_gatedLinear_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Int -> IO ()
_gatedLinear_updateGradInput t0 t1 t2 = Dynamic._gatedLinear_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2)

_gRUFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._gRUFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
_gRUFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_gRUFused_updateGradInput t0 t1 t2 t3 t4 = Dynamic._gRUFused_updateGradInput  (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)

_lSTMFused_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateOutput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateOutput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)
_lSTMFused_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_lSTMFused_updateGradInput t0 t1 t2 t3 t4 t5 t6 = Dynamic._lSTMFused_updateGradInput (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5) (asDynamic t6)


