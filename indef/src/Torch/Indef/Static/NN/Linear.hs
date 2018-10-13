-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Linear
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Linear layers
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Linear where

import Data.List
import GHC.Generics
import Data.Singletons.Prelude.List hiding (All)
import Numeric.Backprop
import Numeric.Dimensions
import System.IO.Unsafe

import Debug.Trace
import Torch.Indef.Types
import Torch.Indef.Static.Tensor
import Torch.Indef.Static.Tensor.Math
import Torch.Indef.Static.Tensor.Math.Blas
import Torch.Indef.Static.Tensor.Math.Pointwise
import Torch.Indef.Static.Tensor.Math.Pointwise.Signed ()
import Torch.Indef.Static.Tensor.Math.Pairwise (Pairwise(..))
import Torch.Indef.Static.NN.Backprop ()
import qualified Torch.Indef.Dynamic.NN as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pairwise as Dynamic

-- | datatype representing a linear layer with bias. Represents
-- @y = Ax + b@.
newtype Linear i o
  = Linear { getTensors :: (Tensor '[i, o], Tensor '[o]) }
  deriving (Eq, Generic)

instance (KnownDim i, KnownDim o) => Show (Linear i o) where
  show c = intercalate ","
    [ "Linear ("
    ++ "input: "  ++ show (inputSize c)
    , " output: " ++ show (outputSize c)
    ++ ")"
    ]

instance (KnownDim i, KnownDim o) => Backprop (Linear i o) where
  zero = const . Linear $ (constant 0, constant 0)
  one  = const . Linear $ (constant 1, constant 1)
-- instance (KnownDim i, KnownDim o) => Backprop (Linear i o) where
--   one  (Linear (a, b)) = unsafePerformIO $ do
--     Dynamic.onesLike_ (asDynamic a) (asDynamic a)
--     Dynamic.onesLike_ (asDynamic b) (asDynamic b)
--     pure (Linear (a, b))
--   {-# NOINLINE one #-}

--   zero (Linear (a, b)) = unsafePerformIO $ do
--     Dynamic.zerosLike_ (asDynamic a) (asDynamic a)
--     Dynamic.zerosLike_ (asDynamic b) (asDynamic b)
--     pure (Linear (a, b))
--   {-# NOINLINE zero #-}


  add (Linear (a0, b0)) (Linear (a1, b1)) = unsafePerformIO $ do
    Dynamic.cadd_ (asDynamic a1) 1 (asDynamic a0)
    Dynamic.cadd_ (asDynamic b1) 1 (asDynamic b0)
    pure (Linear (a1, b1))
  {-# NOINLINE add #-}

instance (KnownDim i, KnownDim o) => Num (Linear i o) where
  (+) (Linear (a0, b0)) (Linear (a1, b1)) = Linear (a0+a1, b0+b1)
  (-) (Linear (a0, b0)) (Linear (a1, b1)) = Linear (a0-a1, b0-b1)
  (*) (Linear (a0, b0)) (Linear (a1, b1)) = Linear (a0*a1, b0*b1)
  abs (Linear (a0, b0)) = Linear (abs a0, abs b0)
  fromInteger i = Linear (fromInteger i, fromInteger i)

instance (KnownDim i, KnownDim o) => Pairwise (Linear i o) HsReal where
  (Linear tens) ^+ v = Linear (tens ^+ v)
  (Linear tens) ^- v = Linear (tens ^+ v)
  (Linear tens) ^* v = Linear (tens ^+ v)
  (Linear tens) ^/ v = Linear (tens ^+ v)

-- -- | update a Linear layer
-- updatePure
--   :: (KnownDim i, KnownDim o)
--   => Linear i o   -- ^ layer to update
--   -> HsReal       -- ^ learning rate
--   -> Linear i o   -- ^ gradient
--   -> Linear i o   -- ^ updated layer
-- updatePure net lr (Linear (gw, gb)) = add net $ Linear (lr *^ gw, lr *^ gb)

-- | update a Conv2d layer
update
  :: (KnownDim i, KnownDim o)
  => Linear i o   -- ^ layer to update
  -> HsReal       -- ^ learning rate
  -> Linear i o   -- ^ gradient
  -> IO ()
update (Linear (w, b)) lr (Linear (gw, gb)) = do
  Dynamic.cadd_ (asDynamic w) lr (asDynamic gw)
  Dynamic.cadd_ (asDynamic b) lr (asDynamic gb)
  pure ()
{-# NOINLINE update #-}


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

mkLinear
  :: (KnownDim i, KnownDim o)
  => (forall d . Dimensions d => IO (Tensor d))
  -> IO (Linear i o)
mkLinear initer = Linear <$> ((,) <$> initer <*> initer)

-- ========================================================================= --

-- | Linear
--
-- module = nn.Linear(inputDimension, outputDimension, [bias = true])
--
-- Applies a linear transformation to the incoming data, i.e. y = Ax + b. The input tensor given in forward(input) must be either a vector (1D tensor) or matrix (2D tensor). If the input is a matrix, then each row is assumed to be an input sample of given batch. The layer can be used without bias by setting bias = false.
--
-- You can create a layer in the following way:
--
--  module = nn.Linear(10, 5)  -- 10 inputs, 5 outputs
--
-- Usually this would be added to a network of some kind, e.g.:
--
--  mlp = nn.Sequential()
--  mlp:add(module)
--
-- The weights and biases (A and b) can be viewed with:
--
--  print(module.weight)
--  print(module.bias)
--
-- The gradients for these weights can be seen with:
--
--  print(module.gradWeight)
--  print(module.gradBias)
--
-- As usual with nn modules, applying the linear transformation is performed with:
--
-- x = torch.Tensor(10) -- 10 inputs
-- y = module:forward(x)
linear
  :: forall s i o
  .  Reifies s W
  => All KnownDim '[i,o]
  => BVar s (Linear i o)
  -> BVar s (Tensor '[i])
  -> BVar s (Tensor '[o])
linear = liftOp2 $ op2 $ \l i ->
  (updateOutput i l, \gout -> (accGradParameters i gout l, updateGradInput i gout (weights l)))
  where
    updateOutput :: Tensor '[i] -> Linear i o -> Tensor '[o]
    updateOutput i (Linear (w,b)) = addmv 1 b 1 (transpose2d w) i

    updateGradInput :: Tensor '[i] -> Tensor '[o] -> Tensor '[i,o] -> Tensor '[i]
    updateGradInput i gout w = addmv 0 (constant 0) 1 w gout

    accGradParameters :: Tensor '[i] -> Tensor '[o] -> Linear i o -> Linear i o
    accGradParameters i gout (Linear (w, b)) = Linear (w', b')
      where
        lr = 1
        w' = addr 1 (constant 0) lr i gout
        b' = cadd b lr gout

-- | 'linear' with a batch dimension
linearBatch
  :: forall s i o b
  .  Reifies s W
  => All KnownDim '[b,i,o]
  => HsReal
  -> BVar s (Linear i o)
  -> BVar s (Tensor '[b, i])
  -> BVar s (Tensor '[b, o])
linearBatch lr = liftOp2 $ op2 $ \l i -> (updateOutput i l, \gout -> (accGradParameters i gout l, updateGradInput i gout (weights l)))
  where
    updateOutput :: Tensor '[b, i] -> Linear i o -> Tensor '[b, o]
    updateOutput i (Linear (w,b)) =
      let
        o = addmm 0 (constant 0) 1 i w
      in
        addr 1 o 1 (constant 1) b

    updateGradInput :: Tensor '[b, i] -> Tensor '[b, o] -> Tensor '[i,o] -> Tensor '[b, i]
    updateGradInput i gout w = addmm 0 (constant 0) 1 gout (transpose2d w)

    accGradParameters :: Tensor '[b,i] -> Tensor '[b,o] -> Linear i o -> Linear i o
    accGradParameters i gout (Linear (w, b)) = Linear (gw, gb) -- addr 1 (constant 0) lr i gout, cadd (constant 0) lr gout)
      where
        gw :: Tensor '[i, o]
        gw = addmm 1 (constant 0) lr (transpose2d i) gout

        gb :: Tensor '[o]
        gb = addmv 1 (constant 0) lr tgout (constant 1)

        tgout :: Tensor '[o,b]
        tgout = transpose2d gout

{-
-- | SparseLinear
--
-- Applies a linear transformation to the incoming sparse data, i.e. y = Ax + b.
-- The input tensor given in forward(input) must be a sparse vector represented
-- as 2D tensor of the form torch.Tensor(N, 2) where the pairs represent indices
-- and values. The SparseLinear layer is useful when the number of input dimensions
-- is very large and the input data is sparse.
--
-- You can create a sparse linear layer in the following way:
--
-- The sparse linear module may be used as part of a larger network, and apart
-- from the form of the input, SparseLinear operates in exactly the same way as
-- the Linear layer.
--
-- A sparse input vector may be created as so...
--
-- > x = torch.Tensor({ {1, 0.1}, {2, 0.3}, {10, 0.3}, {31, 0.2} })
-- > print(x)
-- >   1.0000   0.1000
-- >   2.0000   0.3000
-- >  10.0000   0.3000
-- >  31.0000   0.2000
-- > [torch.Tensor of dimension 4x2]
--
-- The first column contains indices, the second column contains values in a
-- vector where all other elements are zeros. The indices should not exceed the
-- stated dimensions of the input to the layer (10000 in the example).
sparselinear
  :: forall s i o
  .  Reifies s W
  => All KnownDim '[i,o]
  => HsReal
  -> BVar s (Linear i o)
  -> BVar s (Tensor '[i, 2])
  -> BVar s (Tensor '[o])
sparselinear lr = liftOp2 $ op2 $ \l i ->
  let
    o = updateOutput i l
  in
    (o, \gout -> ())
  where
    -- sparseLinear forward pass (updates the output tensor)
    updateOutput :: Tensor '[i, 2] -> Linear i o -> Tensor '[o]
    updateOutput i (Linear (w,b)) = unsafePerformIO $ do
      o <- new
      Dynamic._sparseLinear_updateOutput (asDynamic i) (asDynamic o) (asDynamic w) (asDynamic b)
      pure o

    -- sparseLinear backward-update (updates the layer and bias tensors).
    -- Called 'accGradParameters' in C to indicate accumulating the gradient parameters.
    _accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> IO ()
    _accGradParameters t0 t1 t2 t3 t4 t5 =
      Dynamic._sparseLinear_accGradParameters
        (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4) (asDynamic t5)

    -- sparseLinear zeroGradParameters
    _zeroGradParameters :: Tensor d -> Tensor d -> Tensor d -> IO ()
    _zeroGradParameters t0 t1 t2 =
      Dynamic._sparseLinear_zeroGradParameters (asDynamic t0) (asDynamic t1) (asDynamic t2)

    -- sparseLinear updateParameters
    _updateParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
    _updateParameters t0 t1 t2 t3 t4 =
      Dynamic._sparseLinear_updateParameters (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3) (asDynamic t4)
-}

