-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Compare
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Compare a tensor with a scala value
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Compare
  ( ltValue, ltValueT, ltValueT_
  , leValue, leValueT, leValueT_
  , gtValue, gtValueT, gtValueT_
  , geValue, geValueT, geValueT_
  , neValue, neValueT, neValueT_
  , eqValue, eqValueT, eqValueT_
  ) where

import Foreign hiding (with, new)
import Foreign.Ptr
import Numeric.Dimensions
import System.IO.Unsafe
import Control.Monad.Managed

import Torch.Indef.Types
import Torch.Indef.Mask
import Torch.Indef.Dynamic.Tensor

import qualified Torch.Sig.Tensor.Math.Compare as Sig

_ltValueT, _leValueT, _gtValueT, _geValueT, _neValueT, _eqValueT
  :: Dynamic -> Dynamic -> HsReal -> IO ()
_ltValueT = compareValueTOp Sig.c_ltValueT
_leValueT = compareValueTOp Sig.c_leValueT
_gtValueT = compareValueTOp Sig.c_gtValueT
_geValueT = compareValueTOp Sig.c_geValueT
_neValueT = compareValueTOp Sig.c_neValueT
_eqValueT = compareValueTOp Sig.c_eqValueT

compareValueTOp
  :: (Ptr CState -> Ptr CTensor -> Ptr CTensor -> CReal -> IO ())
  -> Dynamic -> Dynamic -> HsReal -> IO ()
compareValueTOp fn a b v = withLift $ fn
  <$> managedState
  <*> managedTensor a
  <*> managedTensor b
  <*> pure (hs2cReal v)

compareTensorOp
  :: (Ptr CState -> Ptr CByteTensor -> Ptr CTensor -> CReal -> IO ())
  -> Dynamic -> HsReal -> MaskDynamic
compareTensorOp op t0 v = unsafePerformIO . flip with pure $ do
  s' <- managedState
  t' <- managedTensor t0
  SomeDims d <- liftIO $ getDims t0
  let bt = newMaskDyn d
  bt' <- managed $ withMask bt
  liftIO $ op s' bt' t' (hs2cReal v)
  pure bt

{-# NOINLINE compareTensorOp #-}

-- | return a byte tensor which contains boolean values indicating the relation between a tensor and a given scalar.
ltValue, leValue, gtValue, geValue, neValue, eqValue
  :: Dynamic -> HsReal -> MaskDynamic
ltValue = compareTensorOp Sig.c_ltValue
leValue = compareTensorOp Sig.c_leValue
gtValue = compareTensorOp Sig.c_gtValue
geValue = compareTensorOp Sig.c_geValue
neValue = compareTensorOp Sig.c_neValue
eqValue = compareTensorOp Sig.c_eqValue

-- | return a tensor which contains numeric values indicating the relation between a tensor and a given scalar.
-- 0 stands for false, 1 stands for true.
ltValueT, leValueT, gtValueT, geValueT, neValueT, eqValueT
  :: Dynamic -> HsReal -> IO Dynamic
ltValueT  a b = withEmpty a $ \r -> _ltValueT r a b
leValueT  a b = withEmpty a $ \r -> _leValueT r a b
gtValueT  a b = withEmpty a $ \r -> _gtValueT r a b
geValueT  a b = withEmpty a $ \r -> _geValueT r a b
neValueT  a b = withEmpty a $ \r -> _neValueT r a b
eqValueT  a b = withEmpty a $ \r -> _eqValueT r a b

-- | mutate a tensor in-place with its numeric relation to a given scalar, where 0 stands for false and
-- 1 stands for true.
ltValueT_, leValueT_, gtValueT_, geValueT_, neValueT_, eqValueT_
  :: Dynamic -> HsReal -> IO ()
ltValueT_ a b = _ltValueT a a b
leValueT_ a b = _leValueT a a b
gtValueT_ a b = _gtValueT a a b
geValueT_ a b = _geValueT a a b
neValueT_ a b = _neValueT a a b
eqValueT_ a b = _eqValueT a a b

