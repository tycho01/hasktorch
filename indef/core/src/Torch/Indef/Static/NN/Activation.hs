-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.NN.Activation
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.NN.Activation where

import Data.List (intercalate)
import GHC.Generics

import Numeric.Backprop
import Numeric.Dimensions
import Control.Monad
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Internal (new)
import Torch.Indef.Static.Tensor.Math (constant)
import Torch.Indef.Static.Tensor.Copy
import Torch.Indef.Static.NN.Backprop ()

import qualified Torch.Indef.Dynamic.NN.Activation as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pointwise as Dynamic
import qualified Torch.Indef.Dynamic.Tensor.Math.Pairwise as Dynamic

-- | datatype representing a relu activation.
data Relu d = Relu deriving (Eq, Generic)

instance KnownDim d => Show (Relu d) where
  show c = intercalate ","
    [ "Relu ("
    ++ "dimensions: "  ++ show (reluSize c)
    ++ ")"
    ]

-- | The size of a relu
reluSize :: forall d . KnownDim d => Relu d -> Int
reluSize _ = fromIntegral (dimVal (dim :: Dim d))

instance (KnownDim d) => Backprop (Relu d) where
  zero = const Relu
  one  = const Relu
  add Relu Relu = Relu
  {-# NOINLINE add #-}

-- | pReLU updateOutput
_pReLU_updateOutput :: Tensor d -> Tensor d -> Tensor d -> IO ()
_pReLU_updateOutput a0 a1 a2 = Dynamic._pReLU_updateOutput (asDynamic a0) (asDynamic a1) (asDynamic a2)

-- | pReLU updateGradInput
_pReLU_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> IO ()
_pReLU_updateGradInput a0 a1 a2 a3 =
  Dynamic._pReLU_updateGradInput (asDynamic a0) (asDynamic a1) (asDynamic a2) (asDynamic a3)

-- | pReLU accGradParameters
_pReLU_accGradParameters :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> IO ()
_pReLU_accGradParameters a0 a1 a2 a3 a4 =
  Dynamic._pReLU_accGradParameters (asDynamic a0) (asDynamic a1) (asDynamic a2) (asDynamic a3) (asDynamic a4)

-- | rReLU updateOutput
_rReLU_updateOutput :: Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> Bool -> Generator -> IO ()
_rReLU_updateOutput t0 t1 t2 d0 d1 b0 b1 g =
  Dynamic._rReLU_updateOutput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (d1)
    (b0) (b1)
    g

-- | rReLU updateGradInput
_rReLU_updateGradInput   :: Tensor d -> Tensor d -> Tensor d -> Tensor d -> Double -> Double -> Bool -> Bool -> IO ()
_rReLU_updateGradInput t0 t1 t2 t3 d0 d1 b0 b1 =
  Dynamic._rReLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2) (asDynamic t3)
    (d0) (d1)
    (b0) (b1)

-- | eLU updateOutput
_eLU_updateOutput :: Tensor d -> Tensor d -> Double -> Double -> Bool -> IO ()
_eLU_updateOutput t0 t1 d0 d1 b0 =
  Dynamic._eLU_updateOutput
    (asDynamic t0) (asDynamic t1)
    (d0) (d1)
    (b0)

-- | eLU updateGradInput
_eLU_updateGradInput :: Tensor d -> Tensor d' -> Tensor d'' -> Double -> Double -> IO ()
_eLU_updateGradInput t0 t1 t2 d0 d1 =
  Dynamic._eLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (d1)

-- | leakyReLU updateOutput
_leakyReLU_updateOutput :: Tensor d -> Tensor d -> Double -> Bool -> IO ()
_leakyReLU_updateOutput t0 t1 d0 b0 =
  Dynamic._leakyReLU_updateOutput
    (asDynamic t0) (asDynamic t1)
    (d0) (b0)


-- | leakyReLU updateGradInput
_leakyReLU_updateGradInput :: Tensor d -> Tensor d -> Tensor d -> Double -> Bool -> IO ()
_leakyReLU_updateGradInput t0 t1 t2 d0 b0 =
  Dynamic._leakyReLU_updateGradInput
    (asDynamic t0) (asDynamic t1) (asDynamic t2)
    (d0) (b0)

-- | ReLU activation function
relu :: Reifies s W => Dimensions d => BVar s (Tensor d) -> BVar s (Tensor d)
relu = threshold 0 0

-- | ReLU activation function
reluIO :: Dimensions d => Tensor d -> IO (Tensor d, Tensor d -> IO (Tensor d))
reluIO = thresholdIO 0 0


{-# NOINLINE threshold #-}
-- | run a threshold function againts two BVar variables
threshold
  :: forall s d . Reifies s W
  => Dimensions d
  => Double               -- ^ threshold
  -> Double               -- ^ replacement value
  -> BVar s (Tensor d)    -- ^ input
  -> BVar s (Tensor d)    -- ^ output
threshold thr value = liftOp1 . op1 $ \inp -> unsafePerformIO $ do
  (out, getgrad) <- thresholdIO thr value inp
  pure (out, unsafePerformIO . getgrad)

-- | run a threshold function in IO
thresholdIO
  :: forall d
  .  Dimensions d
  => Double      -- ^ threshold
  -> Double      -- ^ replacement value
  -> Tensor d    -- ^ input
  -> IO (Tensor d, Tensor d -> IO (Tensor d))    -- ^ output
thresholdIO thr value inp = do
  out <- _threshold_updateOutput thr value False inp
  pure (out, _threshold_updateGradInput thr value False inp)

  where
    _threshold_updateOutput
      :: Double              -- ^ threshold
      -> Double              -- ^ replacement value
      -> Bool                -- ^ inplace
      -> Tensor d            -- ^ input
      -> IO (Tensor d)       -- ^ output
    _threshold_updateOutput thr val inplace input = do
      let out = new
      -- FIXME: this looks like a bug in ATen. Need to check if this still exists after updating.
      let input' = if inplace then input else copy input

      Dynamic._threshold_updateOutput
        (asDynamic input') (asDynamic out)
        thr val
        inplace

      pure out

    _threshold_updateGradInput
      :: Dimensions d => Double        -- ^ threshold
      -> Double        -- ^ replacement value
      -> Bool          -- ^ inplace
      -> Tensor d      -- ^ input
      -> Tensor d      -- ^ gradient output
      -> IO (Tensor d) -- ^ gradient input
    _threshold_updateGradInput thr val inplace input gout = do
      let gin = new
      let input' = if inplace then input else copy input
      Dynamic._threshold_updateGradInput
        (asDynamic input') (asDynamic gout) (asDynamic gin)
        thr val
        inplace
      pure gin


