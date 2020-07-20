{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Synthesizer.Encoder (module Synthesizer.Encoder) where

import Data.Bifunctor (first, second, bimap)
import Data.Int (Int64) 
import Data.Char (ord)
import Data.HashMap.Lazy (HashMap, (!), toList)
import GHC.Generics (Generic)
import GHC.TypeNats (Nat, KnownNat, type (*))
import Util (fstOf3)

import           Torch.Typed.Tensor
import qualified Torch.Typed.Tensor
import           Torch.Typed.Functional
import           Torch.Typed.Factories
import           Torch.Typed.Aux
import           Torch.Typed.Parameter
import qualified Torch.Typed.Parameter
import           Torch.Autograd
import           Torch.HList
import           Torch.Scalar
import qualified Torch.NN                      as A
import qualified Torch.Functional              as F
import qualified Torch.Functional.Internal     as I
import qualified Torch.Tensor                  as D
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import           Torch.Typed.NN
import           Torch.Typed.NN.Recurrent.Aux
import           Torch.Typed.NN.Recurrent.LSTM

import Synthesis.Orphanage ()
import Synthesis.Data (Expr, Tp, Tpl2)
import Synthesis.Utility (pp, mapBoth, asPairs, safeIndexHM)
import Synthesizer.Utility
import Synthesizer.Params

data LstmEncoderSpec
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (batch_size :: Nat)
    (numChars :: Nat)
    (h :: Nat)
    (featMult :: Nat)
 where LstmEncoderSpec :: {
        charMap :: HashMap Char Int,
        lstmSpec :: LSTMSpec numChars h NumLayers Dir 'D.Float device
    } -> LstmEncoderSpec device maxStringLength batch_size numChars h featMult
 deriving (Show)

data LstmEncoder
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (batch_size :: Nat)
    (numChars :: Nat)
    (h :: Nat)
    (featMult :: Nat)
 where LstmEncoder :: {
    charMap :: HashMap Char Int,
    inModel  :: LSTMWithInit numChars h NumLayers Dir 'ConstantInitialization 'D.Float device,
    outModel :: LSTMWithInit numChars h NumLayers Dir 'ConstantInitialization 'D.Float device
    } -> LstmEncoder device maxStringLength batch_size numChars h featMult
 deriving (Show, Generic)

-- instance (Scalar a) => A.Parameterized a where
instance A.Parameterized Int where
  flattenParameters _ = []
  replaceOwnParameters = return

instance A.Parameterized (LstmEncoder device maxStringLength batch_size numChars h featMult)

instance (KnownDevice device, RandDTypeIsValid device 'D.Float, KnownNat numChars, KnownNat h) => A.Randomizable (LstmEncoderSpec device maxStringLength batch_size numChars h featMult) (LstmEncoder device maxStringLength batch_size numChars h featMult) where
    sample LstmEncoderSpec {..} = do
        in_model  :: LSTMWithInit numChars h NumLayers Dir 'ConstantInitialization 'D.Float device <- A.sample spec
        out_model :: LSTMWithInit numChars h NumLayers Dir 'ConstantInitialization 'D.Float device <- A.sample spec
        return $ LstmEncoder charMap in_model out_model
            -- TODO: consider LearnedInitialization
            where spec :: LSTMWithInitSpec numChars h NumLayers Dir 'ConstantInitialization 'D.Float device = LSTMWithZerosInitSpec lstmSpec

patchEncoderLoss
    :: forall batch_size maxStringLength numChars n' device h featTnsr featMult
     . (KnownDevice device, KnownNat batch_size, KnownNat maxStringLength, KnownNat numChars, KnownNat h, KnownNat featMult, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float)
    => LstmEncoder device maxStringLength batch_size numChars h featMult
    -> Tensor device 'D.Float '[]
    -> Tensor device 'D.Float '[]
patchEncoderLoss encoder_model = let
        dropoutOn = True
        in_dummy  :: Tensor device 'D.Float '[] = mulScalar (0.0 :: Float) $ sumAll $ fstOf3 . lstmDynamicBatch @'SequenceFirst dropoutOn (inModel  encoder_model) $ (ones :: Tensor device 'D.Float '[1,1,numChars])
        out_dummy :: Tensor device 'D.Float '[] = mulScalar (0.0 :: Float) $ sumAll $ fstOf3 . lstmDynamicBatch @'SequenceFirst dropoutOn (outModel encoder_model) $ (ones :: Tensor device 'D.Float '[1,1,numChars])
    in add $ Torch.Typed.Tensor.toDevice $ in_dummy `add` out_dummy
