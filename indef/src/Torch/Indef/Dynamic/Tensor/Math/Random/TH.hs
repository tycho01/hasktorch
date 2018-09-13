-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Random.TH
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
--
-- Random functions for CPU-based tensors.
-------------------------------------------------------------------------------


module Torch.Indef.Dynamic.Tensor.Math.Random.TH
  ( _rand
  , _randn
  , _randperm
  ) where

import Foreign
import Foreign.Ptr
import Control.Monad
import Control.Monad.Managed
import Torch.Indef.Types
import qualified Torch.Types.TH as TH
import qualified Torch.Sig.Types as Sig
import qualified Torch.Sig.Types.Global as Sig
import qualified Torch.Sig.Tensor.Math.Random.TH as Sig

go
  :: (Ptr CState -> Ptr CTensor -> Ptr CGenerator -> Ptr TH.CLongStorage -> IO ())
  -> Dynamic
  -> Generator
  -> TH.IndexStorage
  -> IO ()
go fn d g i = runManaged . (liftIO =<<) $ fn
  <$> managedState
  <*> managedTensor d
  <*> managed (withForeignPtr (Sig.rng g))
  <*> managed (withForeignPtr . snd . TH.longStorageState $ i)

-- | C-style, impure, call to Torch's @rand@ function.
--
-- Returns a tensor filled with random numbers from a uniform distribution
-- on the interval [0, 1).
--
-- In Lua, this is @y = torch.rand(gen, torch.LongStorage{m, n, k, l, o})@, where
-- the 'IndexStorage' holds a list of dimensions to be filled.
_rand  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
_rand = go Sig.c_rand

-- | C-style, impure, call to Torch's @randn@ function.
--
-- Returns a tensor filled with random numbers from a normal distribution with
-- mean zero and variance one.
--
-- In Lua, this is @y = torch.randn(gen, torch.LongStorage{m, n, k, l, o})@, where
-- the 'IndexStorage' holds a list of dimensions to be filled.
_randn  :: Dynamic -> Generator -> TH.IndexStorage -> IO ()
_randn = go Sig.c_randn

-- | C-style, impure, call to Torch's @randperm@ function.
--
-- Returns a random permutation of integers from 1 to @n@
_randperm
  :: Dynamic   -- ^ tensor to mutate, inplace
  -> Generator -- ^ local generator to use
  -> Integer   -- ^ @n@
  -> IO ()
_randperm t g i = runManaged . (liftIO =<<) $ Sig.c_randperm
  <$> managedState
  <*> managedTensor t
  <*> managed (withForeignPtr (Sig.rng g))
  <*> pure (fromIntegral i)


