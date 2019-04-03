-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Floating where

import Control.Monad.Managed
import GHC.Int
import Numeric.Dimensions
import System.IO.Unsafe

import qualified Torch.Sig.Tensor.Math.Floating as Sig
import Torch.Indef.Dynamic.Tensor (empty)

import Torch.Indef.Types

-- | Returns a one-dimensional Tensor of @n@ equally spaced points between @x1@ and @x2@.
linspace
  :: HsReal  -- ^ @x1@
  -> HsReal  -- ^ @x2@
  -> Int64   -- ^ @n@
  -> Dynamic
linspace a b l = unsafeDupablePerformIO $ let t = empty in linspace_ t a b l >> pure t
{-# NOINLINE linspace #-}

-- | inplace version of 'linspace'
linspace_ :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
linspace_ r a b l = runManaged $ do
  s' <- managedState
  r' <- managedTensor r
  liftIO $ Sig.c_linspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)

-- | Returns a one-dimensional Tensor of @n@ logarithmically, equally spaced points between @10^x1@ and @10^x2@.
logspace
  :: HsReal  -- ^ @x1@
  -> HsReal  -- ^ @x2@
  -> Int64   -- ^ @n@
  -> Dynamic
logspace a b l = unsafeDupablePerformIO $ let t = empty in logspace_ t a b l >> pure t
{-# NOINLINE logspace #-}

-- | inplace version of 'logspace'
logspace_ :: Dynamic -> HsReal -> HsReal -> Int64 -> IO ()
logspace_ r a b l = runManaged $ do
  s' <- managedState
  r' <- managedTensor r
  liftIO $ Sig.c_logspace s' r' (hs2cReal a) (hs2cReal b) (fromIntegral l)


