-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.TopK
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Static.Tensor.TopK where

import Numeric.Dimensions
import System.IO.Unsafe

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Internal (new)
import qualified Torch.Indef.Dynamic.Tensor.TopK as Dynamic
import Torch.Indef.Index

-- | returns all @k@ smallest elements in a tensor over a given dimension, including their indices, in unsorted order.
topk
  :: forall d' d n
  .  (All Dimensions '[d, d'], KnownDim n)
  => Tensor d -> Integer -> Word -> TopKOrder -> Maybe KeepDim -> (Tensor d', IndexTensor '[n])
topk t k d o sorted = unsafeDupablePerformIO $ do
  let ix :: IndexTensor '[n] = newIx
  let r  :: Tensor d' = new
  Dynamic._topk (asDynamic r, longAsDynamic ix) (asDynamic t) k d o sorted
  pure (r, ix)
{-# NOINLINE topk #-}


