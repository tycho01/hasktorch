-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.ScatterGather
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.ScatterGather where

import Torch.Indef.Types
import qualified Torch.Indef.Dynamic.Tensor.ScatterGather as Dynamic

-- | Static call to 'Dynamic._gather'
_gather :: Tensor d -> Tensor d -> Word -> IndexTensor '[n] -> IO ()
_gather r src d ix = Dynamic._gather (asDynamic r) (asDynamic src) d (longAsDynamic ix)

-- | Static call to 'Dynamic._scatter'
_scatter :: Tensor d -> Word -> IndexTensor '[n] -> Tensor d -> IO ()
_scatter r d ix src = Dynamic._scatter (asDynamic r) d (longAsDynamic ix) (asDynamic src)

-- | Static call to 'Dynamic._scatterAdd'
_scatterAdd   :: Tensor d -> Word -> IndexTensor '[n] -> Tensor d -> IO ()
_scatterAdd r d ix src = Dynamic._scatterAdd (asDynamic r) d (longAsDynamic ix) (asDynamic src)

-- | Static call to 'Dynamic._scatterFill'
_scatterFill  :: Tensor d -> Word -> IndexTensor '[n] -> HsReal -> IO ()
_scatterFill r d ix = Dynamic._scatterFill (asDynamic r) d (longAsDynamic ix)

