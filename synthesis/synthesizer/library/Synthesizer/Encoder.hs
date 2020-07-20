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

-- instance (KnownDevice device, KnownNat batch_size, KnownNat n', KnownNat maxStringLength, KnownNat numChars, KnownNat h, shape ~ '[n', maxStringLength * (2 * featMult * Dirs * h)])
--     => HasForward (LstmEncoder device maxStringLength batch_size numChars h featMult) [(Expr, Either String Expr)] (Tensor device 'D.Float shape) where
--         forward      = lstmEncoder
--         -- forwardStoch = lstmEncoder

lstmBatch
    :: forall batch_size maxStringLength numChars device h featMult
     . (KnownNat batch_size, KnownNat maxStringLength, KnownNat numChars, KnownNat h, KnownNat featMult)
    => LstmEncoder device maxStringLength batch_size numChars h featMult
    -> Tensor device 'D.Float '[batch_size, featMult * maxStringLength, numChars]
    -> Tensor device 'D.Float '[batch_size, featMult * maxStringLength, numChars]
    -> Tensor device 'D.Float '[batch_size, maxStringLength * (2 * featMult * Dirs * h)]
lstmBatch LstmEncoder{..} in_vec out_vec = feat_vec where
    lstm' = \model -> fstOf3 . lstmForwardWithDropout @'BatchFirst model
    emb_in  :: Tensor device 'D.Float '[batch_size, featMult * maxStringLength, h * Dirs] = lstm'  inModel  in_vec
    emb_out :: Tensor device 'D.Float '[batch_size, featMult * maxStringLength, h * Dirs] = lstm' outModel out_vec
    -- | For each pair, it then concatenates the topmost hidden representation at every time step to produce a 4HT-dimensional feature vector per I/O pair
    feat_vec :: Tensor device 'D.Float '[batch_size, maxStringLength * (2 * featMult * Dirs * h)] =
            -- reshape $ cat @2 $ emb_in :. emb_out :. HNil
            asUntyped (D.reshape [natValI @batch_size, natValI @maxStringLength * (2 * natValI @featMult * natValI @Dirs * natValI @h)]) $ cat @2 $ emb_in :. emb_out :. HNil

-- | NSPS paper's Baseline LSTM encoder
lstmEncoder
    :: forall batch_size maxStringLength numChars n' device h featTnsr featMult
     . (KnownDevice device, KnownNat batch_size, KnownNat maxStringLength, KnownNat numChars, KnownNat h, KnownNat featMult, featTnsr ~ Tensor device 'D.Float '[maxStringLength, numChars])
    => LstmEncoder device maxStringLength batch_size numChars h featMult
    -> HashMap (Tp, Tp) [(Expr, Either String Expr)]
    -> Tensor device 'D.Float '[n', maxStringLength * (2 * featMult * Dirs * h)]
lstmEncoder encoder tp_io_pairs = UnsafeMkTensor feat_vec where
    LstmEncoder{..} = encoder
    maxStringLength_ :: Int = natValI @maxStringLength
    batch_size_ :: Int = natValI @batch_size
    max_char :: Int = natValI @numChars

    -- TODO: use tree encoding (R3NN) also for expressions instead of just converting to string
    str_map :: HashMap (Tpl2 String) [(Tpl2 String)] =
            bimap (mapBoth pp) (fmap (bimap pp (show . second pp))) `asPairs` tp_io_pairs
    -- convert char to one-hot encoding (byte -> 256 1/0s as float) as third lstm dimension
    str2tensor :: Int -> String -> featTnsr =
            \len -> Torch.Typed.Tensor.toDType @'D.Float . UnsafeMkTensor . D.toDevice (deviceVal @device) . (`I.one_hot` max_char) . D.asTensor . padRight 0 len . fmap ((fromIntegral :: Int -> Int64) . (+1) . safeIndexHM charMap)

    both2t :: Tpl2 String -> Tpl2 featTnsr = mapBoth $ str2tensor maxStringLength_
    addTypes :: (featTnsr, [featTnsr]) -> D.Tensor =
        \(tp, vecs) -> let sample_vec = stack' 0 (toDynamic <$> vecs) in sample_vec
    tp_ios :: [(Tpl2 featTnsr, [Tpl2 featTnsr])] = (bimap both2t $ fmap both2t) <$> toList str_map
    vec_pairs :: [(D.Tensor, D.Tensor)] = (\((in_tp, out_tp), ios) -> let (ins, outs) = unzip ios in addTypes `mapBoth` ((in_tp, ins), (out_tp, outs))) <$> tp_ios
    (in_vecs, out_vecs) :: (Tpl2 [Tensor device 'D.Float '[batch_size, featMult * maxStringLength, numChars]]) =
            mapBoth (fmap UnsafeMkTensor . batchTensor batch_size_ . F.cat (F.Dim 0)) . unzip $ vec_pairs
    feat_vecs :: [Tensor device 'D.Float '[batch_size, maxStringLength * (2 * featMult * Dirs * h)]] =
            uncurry (lstmBatch encoder) <$> zip in_vecs out_vecs
    feat_vec :: D.Tensor = F.cat (F.Dim 0) $ toDynamic <$> feat_vecs

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
