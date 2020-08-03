{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Synthesizer.TypedMLP (
    module Synthesizer.TypedMLP
) where

import           Prelude hiding ( tanh )
import           GHC.Generics
import           GHC.TypeLits
import Torch.Typed hiding (Device)

data TypedMLPSpec (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
             (dtype :: DType)
             (device :: (DeviceType, Nat))
  = TypedMLPSpec

data TypedMLP (inputFeatures :: Nat) (outputFeatures :: Nat) (hiddenFeatures :: Nat)
         (dtype :: DType)
         (device :: (DeviceType, Nat))
  = TypedMLP { layer0 :: Linear inputFeatures  hiddenFeatures dtype device
        , layer1 :: Linear hiddenFeatures hiddenFeatures dtype device
        , layer2 :: Linear hiddenFeatures outputFeatures dtype device
        } deriving (Show, Generic)

instance
  (StandardFloatingPointDTypeValidation device dtype) => HasForward
    (TypedMLP inputFeatures outputFeatures hiddenFeatures dtype device)
    (Tensor device dtype '[batchSize, inputFeatures])
    (Tensor device dtype '[batchSize, outputFeatures])
 where
  forward TypedMLP {..} = forward layer2 . tanh . forward layer1 . tanh . forward layer0

instance
  ( KnownDevice device
  , KnownDType dtype
  , All KnownNat '[inputFeatures, outputFeatures, hiddenFeatures]
  , RandDTypeIsValid device dtype
  ) => Randomizable
    (TypedMLPSpec inputFeatures outputFeatures hiddenFeatures dtype device)
    (TypedMLP     inputFeatures outputFeatures hiddenFeatures dtype device)
 where
  sample TypedMLPSpec =
    TypedMLP <$> sample LinearSpec <*> sample LinearSpec <*> sample LinearSpec
