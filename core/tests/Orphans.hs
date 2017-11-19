{-# LANGUAGE ScopedTypeVariables #-}
module Orphans where

import Test.QuickCheck

import Torch.Core.Tensor.Types (TensorDim(..))

instance (Ord a, Num a, Arbitrary a) => Arbitrary (TensorDim a) where
  arbitrary = do
    n::Int <- choose (0, 4)
    case n of
      0 -> pure D0
      1 -> D1 <$> suchThat arbitrary predicate
      2 -> D2 <$> suchThat arbitrary (\(x, y) -> all predicate [x,y])
      3 -> D3 <$> suchThat arbitrary (\(x, y, z) -> all predicate [x,y,z])
      4 -> D4 <$> suchThat arbitrary (\(x, y, z, q) -> all predicate [x,y,z,q])
      _ -> error "impossible: check arbitrary instance boundary"
   where
    -- torch will not deallocate some(?) tensors, which can tank tests
    predicate :: Num a => a -> Bool
    predicate = (< 7)


