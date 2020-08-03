{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Synthesizer.NSPS (module Synthesizer.NSPS) where

import           System.Random                 (StdGen, mkStdGen)
import           System.Timeout                (timeout)
import           System.Directory              (createDirectoryIfMissing)
import           System.CPUTime
import           Text.Printf
import           Data.Maybe                    (fromMaybe)
import           Data.Set                      (Set, empty, insert)
import qualified Data.Set
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Internal      as BS
import qualified Data.ByteString.Lazy.Internal as BL
import           Data.HashMap.Lazy             (HashMap, (!), elems, keys, size, mapWithKey, filterWithKey)
import qualified Data.Csv as Csv
import           Data.Text.Prettyprint.Doc (pretty)
import           Text.Printf
import           Foreign.Marshal.Utils         (fromBool)
import           Control.Monad                 (join, replicateM, forM, void, when)
import           Language.Haskell.Exts.Syntax  ( Exp (..) )
import           Prelude                        hiding (abs)
import           Language.Haskell.Interpreter  ( Interpreter, liftIO, lift )
import           GHC.Exts
import           GHC.Generics                  (Generic)
import           GHC.TypeNats                  (KnownNat, Nat, type (*), type (-))
import qualified Torch.Functional.Internal     as I
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Device                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Optim                   as D
import qualified Torch.Serialize               as D
import qualified Torch.Autograd                as D
import qualified Torch.Functional              as F
import qualified Torch.NN                      as A
import           Torch.Typed.NN.Recurrent.LSTM
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.NN
import           Torch.Typed.Parameter
import qualified Torch.Typed.Parameter
import           Torch.Typed.Factories
import           Torch.Typed.Optim
import           Torch.Typed.Functional
import           Torch.Typed.Autograd
import           Torch.Typed.Serialize
import qualified Torch.Distributions.Distribution as Distribution
import qualified Torch.Distributions.Categorical as Categorical

import           Synthesis.Orphanage ()
import           Synthesis.Data hiding (GridSearchConfig(..), EvolutionaryConfig(..))
import           Synthesis.Utility
import           Synthesis.Ast
import           Synthesis.Generation
import           Synthesis.FindHoles
import           Synthesis.Hint
import           Synthesis.Types
import           Synthesizer.Utility
import           Synthesizer.Encoder
import           Synthesizer.TypeEncoder
import           Synthesizer.R3NN
import           Synthesizer.Params
import           Synthesizer.Synthesizer

instance ( KnownDevice device, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, KnownNat m, KnownNat symbols, KnownNat rules, KnownNat maxStringLength, KnownNat encoderBatch, KnownNat r3nnBatch, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat h, KnownNat featMult, KnownNat regularization, KnownNat clip, shape ~ '[r3nnBatch, maxStringLength * (2 * featMult * Dirs * h)], ruleFeats ~ (maxStringLength * m) ) => Synthesizer device shape rules ruleFeats (NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult) regularization clip where

    weightDecay :: NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> Float
    weightDecay = weight_decay

    clipVal :: NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> Float
    clipVal = max_clip

    encode :: NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> HashMap (Tp, Tp) [(Expr, Either String Expr)]
                -> Tensor device 'D.Float shape
    -- | sampling embedded features to guarantee a static output size
    -- | to allow R3NN a fixed number of samples for its LSTMs, I'm sampling the actual features to make up for potentially multiple type instances giving me a variable number of i/o samples.
    -- | I opted to pick sampling with replacement, which both more naturally handles sample sizes exceeding the number of items, while also seeming to match the spirit of mini-batching by providing more stochastic gradients.
    -- | for my purposes, being forced to pick a fixed sample size means simpler programs with few types may potentially be learned more easily than programs with e.g. a greater number of type instances.
    -- | there should be fancy ways to address this like giving more weight to hard programs (/ samples).
    -- sampled_feats :: Tensor device 'D.Float '[r3nnBatch, maxStringLength * (2 * featMult * Dirs * h)]
    encode mdl io_pairs = lstmEncoder @encoderBatch @maxStringLength @encoderChars @r3nnBatch @device @h (encoder mdl) io_pairs

    predict   :: forall num_holes
                 . NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> HashMap String Int
                -> Expr
                -> Tensor device 'D.Float shape
                -> Tensor device 'D.Float '[num_holes, rules]
    predict mdl = runR3nn $ r3nn (mdl :: NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult)

    patchLoss :: NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> HashMap String Int
                -> Tensor device 'D.Float '[]
                -> Tensor device 'D.Float '[]
    patchLoss = patchR3nnLoss . r3nn

    doStep    :: forall optimizer . (D.Optimizer optimizer)
                => NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
                -> optimizer
                -> Tensor device 'D.Float '[]
                -> Tensor device 'D.Float '[]
                -> IO ([A.Parameter], optimizer)
    doStep model optim loss lr = D.runStep' model optim (toDynamic lr) . clipGradients' (clipVal model) . decayWeights' (weightDecay model) params $ D.grad' (toDynamic loss) params
        where params = A.flattenParameters model

nspsSpec :: forall device m symbols maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h rules featMult . (KnownNat rules, KnownNat m, KnownNat symbols, KnownNat rules, KnownNat maxStringLength, KnownNat encoderBatch, KnownNat r3nnBatch, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat h, KnownNat featMult) => TaskFnDataset -> [(String, Expr)] -> Int -> Double -> Float -> Float -> NSPSSpec device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
nspsSpec TaskFnDataset{..} variants r3nnBatch dropoutRate weight_decay_val max_clip_val = spec where
    useTypes = natValI @featMult > 1
    charMap = if useTypes then bothCharMap else exprCharMap
    encoder_spec :: LstmEncoderSpec device maxStringLength encoderBatch encoderChars h featMult =
        LstmEncoderSpec charMap $ LSTMSpec $ DropoutSpec dropoutRate
    type_encoder_spec :: TypeEncoderSpec device maxStringLength typeEncoderChars m =
        TypeEncoderSpec ruleCharMap $ LSTMSpec $ DropoutSpec dropoutRate
    r3nn_spec :: R3NNSpec device m symbols rules maxStringLength r3nnBatch h typeEncoderChars featMult =
        initR3nn variants r3nnBatch dropoutRate ruleCharMap
    spec :: NSPSSpec device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult =
        NSPSSpec encoder_spec type_encoder_spec r3nn_spec weight_decay_val max_clip_val

data NSPSSpec (device :: (D.DeviceType, Nat)) (m :: Nat) (symbols :: Nat) (rules :: Nat) (maxStringLength :: Nat) (encoderBatch :: Nat) (r3nnBatch :: Nat) (encoderChars :: Nat) (typeEncoderChars :: Nat) (h :: Nat) (featMult :: Nat) where
  NSPSSpec :: forall device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
     . { encoderSpec :: LstmEncoderSpec device maxStringLength encoderBatch encoderChars h featMult
       , typeEncoderSpec :: TypeEncoderSpec device maxStringLength typeEncoderChars m
       , r3nnSpec :: R3NNSpec device m symbols rules maxStringLength r3nnBatch h typeEncoderChars featMult
       , weight_decay_val :: Float
       , max_clip_val :: Float
       }
    -> NSPSSpec device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
 deriving (Show)

data NSPS (device :: (D.DeviceType, Nat)) (m :: Nat) (symbols :: Nat) (rules :: Nat) (maxStringLength :: Nat) (encoderBatch :: Nat) (r3nnBatch :: Nat) (encoderChars :: Nat) (typeEncoderChars :: Nat) (h :: Nat) (featMult :: Nat) where
  NSPS :: forall device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
        . { encoder :: LstmEncoder device maxStringLength encoderBatch encoderChars h featMult
          , rule_encoder :: TypeEncoder device maxStringLength typeEncoderChars m
          , r3nn :: R3NN device m symbols rules maxStringLength r3nnBatch h typeEncoderChars featMult
          , weight_decay :: Float
          , max_clip :: Float
          }
       -> NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult
 deriving (Show, Generic)

instance ( KnownNat m, KnownNat symbols, KnownNat rules, KnownNat maxStringLength, KnownNat encoderBatch, KnownNat r3nnBatch, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat h, KnownNat featMult )
  => A.Parameterized (NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult)

instance ( KnownDevice device, RandDTypeIsValid device 'D.Float, KnownNat m, KnownNat symbols, KnownNat rules, KnownNat maxStringLength, KnownNat encoderBatch, KnownNat r3nnBatch, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat h, KnownNat featMult )
  => A.Randomizable (NSPSSpec device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult) (NSPS device m symbols rules maxStringLength encoderBatch r3nnBatch encoderChars typeEncoderChars h featMult) where
    sample NSPSSpec {..} = do
        debug_ $ printf
                "NSPS device m=%03d symbols=%03d rules=%03d maxStringLength=%04d encoderBatch=%03d r3nnBatch=%03d encoderChars=%03d typeEncoderChars=%03d h=%03d featMult=%03d\n"
                (natValI @m)
                (natValI @symbols)
                (natValI @rules)
                (natValI @maxStringLength)
                (natValI @encoderBatch)
                (natValI @r3nnBatch)
                (natValI @encoderChars)
                (natValI @typeEncoderChars)
                (natValI @h)
                (natValI @featMult)
        encoder     <- A.sample encoderSpec
        r3nn        <- A.sample r3nnSpec
        return $ NSPS encoder typeEncoder r3nn weight_decay_val max_clip_val
