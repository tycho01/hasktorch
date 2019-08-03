-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Static.Tensor.Random
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
module Torch.Indef.Static.Tensor.Random
  ( random
  , clampedRandom
  , cappedRandom
  , bernoulli
  , bernoulli_DoubleTensor
  , geometric
  , uniform
  , normal
  , normal_means
  , normal_stddevs
  , normal_means_stddevs
  , logNormal
  , exponential
  , cauchy
  , rand
  , randn
  , _multinomial
  , _multinomialAliasSetup
  , _multinomialAliasDraw


  , OpenUnit, openUnit, openUnitValue
  , ClosedUnit, closedUnit, closedUnitValue
  , Positive, positive, positiveValue
  , Ord2Tuple, ord2Tuple, ord2TupleValue
  ) where


import Numeric.Dimensions

import Torch.Indef.Types
import Torch.Indef.Static.Tensor.Internal (new)
import qualified Torch.Indef.Dynamic.Tensor.Random as Dynamic
import qualified Torch.Types.TH as TH
import Torch.Types.Numeric
  ( OpenUnit   , openUnit   , openUnitValue
  , ClosedUnit , closedUnit , closedUnitValue
  , Positive   , positive   , positiveValue
  , Ord2Tuple  , ord2Tuple  , ord2TupleValue
  )


-- | Static call to 'Dynamic.random'
random :: (Dimensions d) => IO (Tensor d)
random = let d = new in Dynamic._random (asDynamic d) >> pure d

-- | Static call to 'Dynamic.clampedRandom'
clampedRandom :: (Dimensions d) => Integer -> Integer -> IO (Tensor d)
clampedRandom a b = let d = new in Dynamic._clampedRandom (asDynamic d) a b >> pure d

-- | Static call to 'Dynamic.cappedRandom'
cappedRandom :: (Dimensions d) => Integer -> IO (Tensor d)
cappedRandom a = let d = new in Dynamic._cappedRandom (asDynamic d) a >> pure d

-- | Static call to 'Dynamic.bernoulli'
bernoulli :: (Dimensions d) => HsAccReal -> IO (Tensor d)
bernoulli a = let d = new in Dynamic._bernoulli (asDynamic d) a >> pure d

-- | Static call to 'Dynamic.geometric'
geometric :: (Dimensions d) => HsAccReal -> IO (Tensor d)
geometric a = let d = new in Dynamic._geometric (asDynamic d) a >> pure d

-- | Static call to 'Dynamic.bernoulli_DoubleTensor'
bernoulli_DoubleTensor :: (Dimensions d) => Tensor d -> IO (Tensor d)
bernoulli_DoubleTensor t = let r = new in Dynamic._bernoulli_DoubleTensor (asDynamic r) (asDynamic t) >> pure r

-- | Static call to 'Dynamic.uniform'
uniform :: Dimensions d => Ord2Tuple HsAccReal -> IO (Tensor d)
uniform tup = let d = new in Dynamic._uniform (asDynamic d) tup >> pure d

-- | Static call to 'Dynamic.normal'
normal :: Dimensions d => HsAccReal -> Positive HsAccReal -> IO (Tensor d)
normal a b = let d = new in Dynamic._normal (asDynamic d) a b >> pure d

-- | Static call to 'Dynamic.normal_means'
normal_means :: (Dimensions d) => Tensor d -> Positive HsAccReal -> IO (Tensor d)
normal_means a b = let d = new in Dynamic._normal_means (asDynamic d) (asDynamic a) b >> pure d

-- | Static call to 'Dynamic.normal_stddevs'
normal_stddevs :: (Dimensions d) => HsAccReal -> Tensor d -> IO (Tensor d)
normal_stddevs a b = let d = new in Dynamic._normal_stddevs (asDynamic d) a (asDynamic b) >> pure d

-- | Static call to 'Dynamic.normal_means_stddevs'
normal_means_stddevs :: (Dimensions d) => Tensor d -> Tensor d -> IO (Tensor d)
normal_means_stddevs a b = let d = new in
  Dynamic._normal_means_stddevs (asDynamic d) (asDynamic a) (asDynamic b) >> pure d

-- | Static call to 'Dynamic.logNormal'
logNormal :: Dimensions d => HsAccReal -> Positive HsAccReal -> IO (Tensor d)
logNormal a b = let d = new in Dynamic._logNormal (asDynamic d) a b >> pure d

-- | Static call to 'Dynamic.exponential'
exponential :: (Dimensions d) => HsAccReal -> IO (Tensor d)
exponential a = let d = new in Dynamic._exponential (asDynamic d) a >> pure d

-- | Static call to 'Dynamic.cauchy'
cauchy :: (Dimensions d) => HsAccReal -> HsAccReal -> IO (Tensor d)
cauchy a b = let d = new in Dynamic._cauchy (asDynamic d) a b >> pure d

-- | Static call to 'Dynamic.rand'
rand :: (Dimensions d) => TH.LongStorage -> IO (Tensor d)
rand a = let d = new in Dynamic._rand (asDynamic d) a >> pure d

-- | Static call to 'Dynamic.randn'
randn :: (Dimensions d) => TH.LongStorage -> IO (Tensor d)
randn a = let d = new in Dynamic._randn (asDynamic d) a >> pure d

-- | Static call to 'Dynamic._multinomial'
_multinomial d t = Dynamic._multinomial (longAsDynamic d) (asDynamic t)

-- | Static call to 'Dynamic._multinomialAliasSetup'
_multinomialAliasSetup d l t = Dynamic._multinomialAliasSetup (asDynamic d) (longAsDynamic l) (asDynamic t)

-- | Static call to 'Dynamic._multinomialAliasDraw'
_multinomialAliasDraw d a b = Dynamic._multinomialAliasDraw (longAsDynamic d) (longAsDynamic a) (asDynamic b)
