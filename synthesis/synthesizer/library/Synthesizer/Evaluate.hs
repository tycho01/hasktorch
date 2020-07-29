{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

-- | synthesizer logic
module Synthesizer.Evaluate (module Synthesizer.Evaluate) where

import System.Log.Logger
import Text.Printf
import GHC.TypeNats (type (+))
import GHC.TypeLits
import Data.Yaml
import Data.HashMap.Lazy (size)
import Control.Monad (void)
import Language.Haskell.Interpreter (Interpreter, liftIO)
import Torch.Internal.Managed.Type.Context (manual_seed_L)
import Torch.Typed.Tensor
import Torch.Typed.Functional
import Torch.Typed.Factories
import Torch.Typed.Aux
import Torch.Typed.NN
import Torch.Typed.NN.Recurrent.LSTM
import qualified Torch.DType                   as D
import qualified Torch.Serialize               as D
import qualified Torch.Autograd                as D
import qualified Torch.NN                      as A
import Synthesis.Data
import Synthesis.Hint
import Synthesis.Orphanage ()
import Synthesis.Data (TaskFnDataset (..), EvaluateConfig (..))
import Synthesis.Configs
import Synthesis.Utility
import Synthesizer.Utility
import Synthesizer.Encoder
import Synthesizer.TypeEncoder
import Synthesizer.R3NN
import Synthesizer.NSPS
import Synthesizer.Params
import Synthesizer.Random
import Synthesizer.Train

-- | main function
main :: IO ()
main = do
    EvaluateConfig{..} <- parseEvaluateConfig
    if gpu
        then synthesize @Gpu
        else synthesize @Cpu

synthesize :: forall device . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float) => IO ()
synthesize = do
    cfg :: EvaluateConfig <- parseEvaluateConfig
    say_ $ show cfg
    let EvaluateConfig{..} = cfg
    updateGlobalLogger logger . setLevel $ logPriority verbosity
    taskFnDataset :: TaskFnDataset <- decodeFileThrow taskPath
    let TaskFnDataset{..} = taskFnDataset
    say_ $ show generationCfg
    manual_seed_L $ fromIntegral seed
    (!! length exprBlocks) $
        -- featMult
        if useTypes then
            getRules @device @2 @0 cfg taskFnDataset
        else
            getRules @device @1 @0 cfg taskFnDataset

getRules :: forall device featMult rules . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getRules cfg taskFnDataset = let
        TaskFnDataset{..} = taskFnDataset
        useTypes = natValI @featMult > 1
        charMap = if useTypes then bothCharMap else exprCharMap
    in (:)
        ((!! (size charMap + 2)) $ getEncoderChars @device @featMult @rules @0 cfg taskFnDataset)
        $ getRules @device @featMult @(rules + 1) cfg taskFnDataset

getEncoderChars :: forall device featMult rules encoderChars . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getEncoderChars cfg taskFnDataset = let
    TaskFnDataset{..} = taskFnDataset
    in (:)
        ((!! (size ruleCharMap + 2)) $ getTypeEncoderChars @device @featMult @rules @encoderChars @0 cfg taskFnDataset)
        $ getEncoderChars @device @featMult @rules @(encoderChars + 1) cfg taskFnDataset

getTypeEncoderChars :: forall device featMult rules encoderChars typeEncoderChars . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getTypeEncoderChars cfg taskFnDataset = (:)
        ((!! (size (dsl taskFnDataset) + natValI @LhsSymbols)) $ getSymbols @device @featMult @rules @encoderChars @typeEncoderChars @0 cfg taskFnDataset)
        $ getTypeEncoderChars @device @featMult @rules @encoderChars @(typeEncoderChars + 1) cfg taskFnDataset

getSymbols :: forall device featMult rules encoderChars typeEncoderChars symbols . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getSymbols cfg taskFnDataset = let
        useTypes = natValI @featMult > 1
        longest = if useTypes then longestString else longestExprString
    in (:)
        ((!! longest taskFnDataset) $ getMaxStringLength @device @featMult @rules @encoderChars @typeEncoderChars @symbols @0 cfg taskFnDataset)
        $ getSymbols @device @featMult @rules @encoderChars @typeEncoderChars @(symbols + 1) cfg taskFnDataset

getMaxStringLength :: forall device featMult rules encoderChars typeEncoderChars symbols maxStringLength . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getMaxStringLength cfg taskFnDataset = let EvaluateConfig{..} = cfg in (:)
        ((!! h) $ getH @device @featMult @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @0 cfg taskFnDataset)
        $ getMaxStringLength @device @featMult @rules @encoderChars @typeEncoderChars @symbols @(maxStringLength + 1) cfg taskFnDataset

getH :: forall device featMult rules encoderChars typeEncoderChars symbols maxStringLength h . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength, KnownNat h) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getH cfg taskFnDataset = let EvaluateConfig{..} = cfg in (:)
        ((!! m) $ getM @device @featMult @0 @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @h cfg taskFnDataset)
        $ getH @device @featMult @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @(h + 1) cfg taskFnDataset

--  shape synthesizer
-- , KnownShape shape, Synthesizer device shape rules synthesizer
getM :: forall device featMult m rules encoderChars typeEncoderChars symbols maxStringLength h . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat m, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength, KnownNat h) => EvaluateConfig -> TaskFnDataset -> [IO ()]
getM cfg taskFnDataset = let
    EvaluateConfig{..} = cfg
    TaskFnDataset{..} = taskFnDataset
    variants :: [(String, Expr)] = (\(_k, v) -> (nodeRule v, v)) <$> exprBlocks
    in (:)
        (do
            let prepped_dsl = prep_dsl taskFnDataset
            let PreppedDSL{..} = prepped_dsl
            let dataset = pickDataset datasets evaluateSet
            (acc, loss) <- case synthesizer of
                "random" -> do
                    model <- A.sample RandomSynthesizerSpec
                    evaluate @device @rules @'[] @0 @RandomSynthesizer taskFnDataset prepped_dsl bestOf maskBad randomHole model dataset
                "nsps" -> do
                    params <- fmap D.IndependentTensor <$> D.load modelPath
                    model <- A.sample spec
                    let model' = A.replaceParameters model params
                    evaluate @device @rules @'[R3nnBatch, maxStringLength * (2 * featMult * Dirs * h)] @(maxStringLength * m) @(NSPS device m symbols rules maxStringLength EncoderBatch R3nnBatch encoderChars typeEncoderChars h featMult) taskFnDataset prepped_dsl bestOf maskBad randomHole model' dataset
                    where
                    useTypes = natValI @featMult > 1
                    charMap = if useTypes then bothCharMap else exprCharMap
                    encoder_spec :: LstmEncoderSpec device maxStringLength EncoderBatch encoderChars h featMult =
                        LstmEncoderSpec charMap $ LSTMSpec $ DropoutSpec dropoutRate
                    r3nn_spec :: R3NNSpec device m symbols rules maxStringLength R3nnBatch h typeEncoderChars featMult =
                        initR3nn variants r3nnBatch dropoutRate ruleCharMap
                    rule_encoder_spec :: TypeEncoderSpec device maxStringLength typeEncoderChars m =
                        TypeEncoderSpec ruleCharMap $ LSTMSpec $ DropoutSpec dropoutRate
                    spec :: NSPSSpec device m symbols rules maxStringLength EncoderBatch R3nnBatch encoderChars typeEncoderChars h featMult =
                        NSPSSpec encoder_spec rule_encoder_spec r3nn_spec
                _ -> error "synthesizer not recognized"
            say_ $ printf
                    "Loss: %.4f. Accuracy: %.4f.\n"
                    loss
                    acc
            )
        $ getM @device @featMult @(m + 1) @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @h cfg taskFnDataset
