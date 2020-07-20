{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Synthesizer.R3NN (module Synthesizer.R3NN) where

import Data.HashMap.Lazy (HashMap, fromList, toList, (!), elems)
import Control.Arrow ((&&&))
import Control.Monad ((=<<), join)
import Language.Haskell.Exts.Syntax
import GHC.Generics
import GHC.TypeNats (KnownNat, Nat, Div, type (*), type (+))
import Util (fstOf3)

import Torch.Typed.Tensor
import Torch.Typed.Functional
import Torch.Typed.NN
import Torch.Typed.Aux
import Torch.Typed.Parameter
import qualified Torch.Typed.Parameter
import Torch.Typed.Factories
import Torch.Autograd
import Torch.Typed.NN
import Torch.Typed.NN.Recurrent.Aux
import Torch.Typed.NN.Recurrent.LSTM
import Torch.HList
import qualified Torch.NN                      as A
import qualified Torch.Functional              as F
import qualified Torch.Tensor                  as D
import qualified Torch.Device                  as D
import qualified Torch.DType                   as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Functional.Internal     as I
import qualified Torch.Autograd                as D

import Synthesis.Orphanage ()
import Synthesis.Data (Expr, Tp)
import Synthesis.Types (holeType)
import Synthesis.FindHoles
import Synthesis.Utility
import Synthesizer.Utility
import Synthesizer.UntypedMLP
import Synthesizer.Params

data R3NNSpec
    (device :: (D.DeviceType, Nat))
    (m :: Nat)
    (symbols :: Nat)
    (rules :: Nat)
    (maxStringLength :: Nat)
    (batch_size :: Nat)
    (h :: Nat)
    (numChars :: Nat)
    (featMult :: Nat)
 where
    R3NNSpec :: forall device m symbols rules maxStringLength batch_size h numChars featMult
        -- symbolIdxs :: HashMap String Int
        -- ppt :: Expr
      . { variant_sizes :: HashMap String Int
        , scoreSpec     :: LSTMSpec  m                                                             (Div m Dirs) NumLayers Dir 'D.Float device
        }
        -> R3NNSpec device m symbols rules maxStringLength batch_size h numChars featMult
 deriving (Show)

data R3NN
    (device :: (D.DeviceType, Nat))
    (m          :: Nat)
    (symbols    :: Nat)
    (rules      :: Nat)
    (maxStringLength :: Nat)
    (batch_size :: Nat)
    (h          :: Nat)
    (numChars    :: Nat)
    (featMult   :: Nat)
    -- I imagine NSPS fixed their batch to the sample size, but I have those for each type instantiation, making this harder for me to fix. as a work-around, I'm sampling instead.
 where
    R3NN :: forall m symbols rules maxStringLength batch_size device h numChars featMult
      . { score_model     :: LSTMWithInit  m                                    (Div m Dirs) NumLayers Dir 'ConstantInitialization 'D.Float device
        , left_nnets :: HashMap String MLP
        , symbol_emb :: Parameter device 'D.Float '[symbols, m]
        }
        -> R3NN device m symbols rules maxStringLength batch_size h numChars featMult
 deriving (Show, Generic)

-- cannot use static Parameterized, as the contents of left_nnets / right_nnets are not statically known in its current HashMap type
instance ( KnownNat m
         , KnownNat symbols
         , KnownNat rules
         , KnownNat maxStringLength
         , KnownNat batch_size
         , KnownNat h
         , KnownNat numChars
         )
  => A.Parameterized (R3NN device m symbols rules maxStringLength batch_size h numChars featMult)

instance ( KnownDevice device
         , RandDTypeIsValid device 'D.Float
         , KnownNat m
         , KnownNat symbols
         , KnownNat rules
         , KnownNat maxStringLength
         , KnownNat batch_size
         , KnownNat h
         , KnownNat numChars
         , KnownNat featMult
         )
  => A.Randomizable (R3NNSpec device m symbols rules maxStringLength batch_size h numChars featMult)
                    (R3NN     device m symbols rules maxStringLength batch_size h numChars featMult)
 where
    sample R3NNSpec {..} = do
        join . return $ R3NN
            <*> A.sample (LSTMWithZerosInitSpec scoreSpec)
            <*> mapM (\q -> A.sample $ MLPSpec (q * m) m) variant_sizes
            <*> (fmap UnsafeMkParameter . D.makeIndependent =<< D.randnIO' [symbols, m])
            where
                -- m must be divisible by Dirs for `Div` in the LSTM specs to work out due to integer division...
                m = assertP ((== 0) . (`mod` natValI @Dirs)) $ natValI @m
                symbols = natValI @symbols

-- | initialize R3NN spec
initR3nn :: forall m symbols rules maxStringLength batch_size h device numChars featMult
         . (KnownNat m, KnownNat symbols, KnownNat rules, KnownNat maxStringLength, KnownNat batch_size, KnownNat h, KnownNat featMult)
         => [(String, Expr)]
         -> Int
         -> Double
         -> HashMap Char Int
         -> (R3NNSpec device m symbols rules maxStringLength batch_size h numChars featMult)
initR3nn variants batch_size dropoutRate charMap = R3NNSpec @device @m @symbols @rules @maxStringLength @batch_size @h @numChars @featMult
        variant_sizes
        (LSTMSpec $ DropoutSpec dropoutRate)
    where
        variant_sizes :: HashMap String Int = fromList $ variantInt . snd <$> variants

variantInt :: Expr -> (String, Int)
variantInt = (appRule &&& length) . fnAppNodes

-- | Torch gets sad not all nnets get used in the loss 😢 so let's give it a hug... 🤗🙄
patchR3nnLoss :: forall m symbols rules maxStringLength batch_size device h numChars featMult . (KnownNat m, KnownNat featMult, KnownNat h, KnownNat batch_size, KnownNat maxStringLength, KnownDevice device, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float) => R3NN device m symbols rules maxStringLength batch_size h numChars featMult -> HashMap String Int -> Tensor device 'D.Float '[] -> Tensor device 'D.Float '[]
patchR3nnLoss r3nn_model variant_sizes = let
        dropoutOn = True
        m :: Int = natValI @m
        left_dummy  :: Tensor device 'D.Float '[] = mulScalar (0.0 :: Float) $ sumAll $ Torch.Typed.Tensor.toDType @'D.Float . UnsafeMkTensor $ F.cat (F.Dim 1) $ fmap (\(k,mlp_) -> let q = safeIndexHM variant_sizes k in mlp mlp_ $ D.zeros' [1,q*m]) $ toList $  left_nnets r3nn_model
        score_dummy :: Tensor device 'D.Float '[] = mulScalar (0.0 :: Float) $ sumAll $ fstOf3 . lstmDynamicBatch @'SequenceFirst dropoutOn (score_model r3nn_model) $ (ones :: Tensor device 'D.Float '[1,1,m])
        symbol_dummy :: Tensor device 'D.Float '[] = mulScalar (0.0 :: Float) $ sumAll $ Torch.Typed.Parameter.toDependent $ (symbol_emb r3nn_model)
    in add $ Torch.Typed.Tensor.toDevice $ left_dummy `add` score_dummy `add` symbol_dummy
