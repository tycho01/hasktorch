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

-- | adjusted from Encoder.hs
module Synthesizer.TypeEncoder (module Synthesizer.TypeEncoder) where

import Data.Bifunctor (second, bimap)
import Data.Int (Int64) 
import Data.Char (ord)
import Data.HashMap.Lazy (HashMap, (!), toList)
import GHC.Generics (Generic)
import GHC.TypeNats (Nat, KnownNat, Div, type (*))
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
import Synthesizer.Encoder

data TypeEncoderSpec
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (numChars :: Nat)
    (m :: Nat)
 where TypeEncoderSpec :: {
        charMap :: HashMap Char Int,
        lstmSpec :: LSTMSpec numChars (Div m Dirs) NumLayers Dir 'D.Float device
    } -> TypeEncoderSpec device maxStringLength numChars m
 deriving (Show)

data TypeEncoder
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (numChars :: Nat)
    (m :: Nat)
 where TypeEncoder :: {
    charMap :: HashMap Char Int,
    ruleModel :: LSTMWithInit numChars (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device
    } -> TypeEncoder device maxStringLength numChars m
 deriving (Show, Generic)

instance A.Parameterized (TypeEncoder device maxStringLength numChars m)

instance (KnownDevice device, RandDTypeIsValid device 'D.Float, KnownNat numChars, KnownNat m) => A.Randomizable (TypeEncoderSpec device maxStringLength numChars m) (TypeEncoder device maxStringLength numChars m) where
    sample TypeEncoderSpec {..} = do
        rule_model  :: LSTMWithInit numChars (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device <- A.sample spec
        return $ TypeEncoder charMap rule_model
            where spec :: LSTMWithInitSpec numChars (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device = LSTMWithZerosInitSpec lstmSpec

typeEncoder
    :: forall batch_size maxStringLength numChars device m featTnsr
     . (KnownDevice device, KnownNat maxStringLength, KnownNat numChars, KnownNat m, featTnsr ~ Tensor device 'D.Float '[1, maxStringLength, numChars])
    => TypeEncoder device maxStringLength numChars m
    -> [Tp]
    -> Tensor device 'D.Float '[batch_size, maxStringLength * m]
typeEncoder TypeEncoder{..} types = feat_vec where
    maxStringLength_ :: Int = natValI @maxStringLength
    max_char :: Int = natValI @numChars
    strs :: [String] = pp <$> types
    str2tensor :: Int -> String -> featTnsr =
        \len -> Torch.Typed.Tensor.toDType @'D.Float . UnsafeMkTensor . D.toDevice (deviceVal @device) . flip I.one_hot max_char . D.asTensor . padRight 0 len . fmap ((fromIntegral :: Int -> Int64) . (+1) . safeIndexHM charMap)
    vecs :: [featTnsr] = str2tensor maxStringLength_ <$> strs
    mdl_vec :: Tensor device 'D.Float '[batch_size, maxStringLength, numChars] =
            UnsafeMkTensor . stack' 0 $ toDynamic <$> vecs
    emb_mdl :: Tensor device 'D.Float '[batch_size, maxStringLength, m] =
        -- asUntyped to type-check m*2/2
        asUntyped id .
        fstOf3 . lstmDynamicBatch @'BatchFirst dropoutOn ruleModel $ mdl_vec
            where dropoutOn = True
    feat_vec :: Tensor device 'D.Float '[batch_size, maxStringLength * m] =
            asUntyped (D.reshape [-1, natValI @maxStringLength * natValI @m]) $ emb_mdl
