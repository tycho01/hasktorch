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
import Synthesis.Hint
import Synthesis.Data (Expr, Tp, Tpl2)
import Synthesis.Utility (pp, mapBoth, asPairs, safeIndexHM)
import Synthesizer.Utility
import Synthesizer.Params
import Synthesizer.Encoder

data TypeEncoderSpec
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (maxChar :: Nat)
    (m :: Nat)
 where TypeEncoderSpec :: {
        charMap :: HashMap Char Int,
        lstmSpec :: LSTMSpec maxChar (Div m Dirs) NumLayers Dir 'D.Float device
    } -> TypeEncoderSpec device maxStringLength maxChar m
 deriving (Show)

data TypeEncoder
    (device :: (D.DeviceType, Nat))
    (maxStringLength :: Nat)
    (maxChar :: Nat)
    (m :: Nat)
 where TypeEncoder :: {
    charMap :: HashMap Char Int,
    ruleModel :: LSTMWithInit maxChar (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device
    } -> TypeEncoder device maxStringLength maxChar m
 deriving (Show, Generic)

instance A.Parameterized (TypeEncoder device maxStringLength maxChar m)

instance (KnownDevice device, RandDTypeIsValid device 'D.Float, KnownNat maxChar, KnownNat m) => A.Randomizable (TypeEncoderSpec device maxStringLength maxChar m) (TypeEncoder device maxStringLength maxChar m) where
    sample TypeEncoderSpec {..} = do
        rule_model  :: LSTMWithInit maxChar (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device <- A.sample spec
        return $ TypeEncoder charMap rule_model
            where spec :: LSTMWithInitSpec maxChar (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device = LSTMWithZerosInitSpec lstmSpec

typeEncoder
    :: forall batch_size maxStringLength maxChar device m featTnsr
     . (KnownDevice device, KnownNat maxStringLength, KnownNat maxChar, KnownNat m, featTnsr ~ Tensor device 'D.Float '[1, maxStringLength, maxChar])
    => TypeEncoder device maxStringLength maxChar m
    -> [Tp]
    -> IO (Tensor device 'D.Float '[batch_size, maxStringLength * m])
typeEncoder TypeEncoder{..} types = do
    debug_ $ "typeEncoder"
    let maxStringLength_ :: Int = natValI @maxStringLength
    let max_char :: Int = natValI @maxChar
    let strs :: [String] = pp <$> types
    debug_ $ "strs: " <> show strs
    let str2tensor :: Int -> String -> featTnsr =
        \len -> Torch.Typed.Tensor.toDType @'D.Float . UnsafeMkTensor . D.toDevice (deviceVal @device) . flip I.one_hot max_char . D.asTensor . padRight 0 len . fmap ((fromIntegral :: Int -> Int64) . (+1) . safeIndexHM charMap)
    let vecs :: [featTnsr] = str2tensor maxStringLength_ <$> strs
    debug_ $ "vecs: " <> show (shape' vecs)
    let mdl_vec :: Tensor device 'D.Float '[batch_size, maxStringLength, maxChar] =
            UnsafeMkTensor . stack' 0 $ toDynamic <$> vecs
    let emb_mdl :: Tensor device 'D.Float '[batch_size, maxStringLength, m] =
        -- asUntyped to type-check m*2/2
        asUntyped id .
        fstOf3 . lstmDynamicBatch @'BatchFirst dropoutOn ruleModel $ mdl_vec
            where dropoutOn = True
    let feat_vec :: Tensor device 'D.Float '[batch_size, maxStringLength * m] =
            asUntyped (D.reshape [-1, natValI @maxStringLength * natValI @m]) $ emb_mdl
    return feat_vec
