{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

-- | synthesizer logic
module Synthesizer.Synthesize (module Synthesizer.Synthesize) where

import System.Log.Logger
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
import Synthesis.Data (TaskFnDataset (..), SynthesizerConfig (..))
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
    SynthesizerConfig{..} <- parseSynthesizerConfig
    if gpu
        then synthesize @Gpu
        else synthesize @Cpu

synthesize :: forall device . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float) => IO ()
synthesize = do
    cfg :: SynthesizerConfig <- parseSynthesizerConfig
    say_ $ show cfg
    let SynthesizerConfig{..} = cfg
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

getRules :: forall device featMult rules . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getRules cfg taskFnDataset = let
        TaskFnDataset{..} = taskFnDataset
        useTypes = natValI @featMult > 1
        charMap = if useTypes then bothCharMap else exprCharMap
    in (:)
        ((!! (size charMap + 2)) $ getEncoderChars @device @featMult @rules @0 cfg taskFnDataset)
        $ getRules @device @featMult @(rules + 1) cfg taskFnDataset

getEncoderChars :: forall device featMult rules encoderChars . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getEncoderChars cfg taskFnDataset = let
    TaskFnDataset{..} = taskFnDataset
    in (:)
        ((!! (size ruleCharMap + 2)) $ getTypeEncoderChars @device @featMult @rules @encoderChars @0 cfg taskFnDataset)
        $ getEncoderChars @device @featMult @rules @(encoderChars + 1) cfg taskFnDataset

getTypeEncoderChars :: forall device featMult rules encoderChars typeEncoderChars . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getTypeEncoderChars cfg taskFnDataset = (:)
        ((!! (size (dsl taskFnDataset) + natValI @LhsSymbols)) $ getSymbols @device @featMult @rules @encoderChars @typeEncoderChars @0 cfg taskFnDataset)
        $ getTypeEncoderChars @device @featMult @rules @encoderChars @(typeEncoderChars + 1) cfg taskFnDataset

getSymbols :: forall device featMult rules encoderChars typeEncoderChars symbols . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getSymbols cfg taskFnDataset = let
        useTypes = natValI @featMult > 1
        longest = if useTypes then longestString else longestExprString
    in (:)
        ((!! longest taskFnDataset) $ getMaxStringLength @device @featMult @rules @encoderChars @typeEncoderChars @symbols @0 cfg taskFnDataset)
        $ getSymbols @device @featMult @rules @encoderChars @typeEncoderChars @(symbols + 1) cfg taskFnDataset

getMaxStringLength :: forall device featMult rules encoderChars typeEncoderChars symbols maxStringLength . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getMaxStringLength cfg taskFnDataset = let SynthesizerConfig{..} = cfg in (:)
        ((!! h) $ getH @device @featMult @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @0 cfg taskFnDataset)
        $ getMaxStringLength @device @featMult @rules @encoderChars @typeEncoderChars @symbols @(maxStringLength + 1) cfg taskFnDataset

getH :: forall device featMult rules encoderChars typeEncoderChars symbols maxStringLength h . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength, KnownNat h) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getH cfg taskFnDataset = let SynthesizerConfig{..} = cfg in (:)
        ((!! m) $ getM @device @featMult @0 @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @h cfg taskFnDataset)
        $ getH @device @featMult @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @(h + 1) cfg taskFnDataset

--  shape synthesizer
-- , KnownShape shape, Synthesizer device shape rules synthesizer
getM :: forall device featMult m rules encoderChars typeEncoderChars symbols maxStringLength h . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat featMult, KnownNat m, KnownNat rules, KnownNat encoderChars, KnownNat typeEncoderChars, KnownNat symbols, KnownNat maxStringLength, KnownNat h) => SynthesizerConfig -> TaskFnDataset -> [IO ()]
getM cfg taskFnDataset = let
    SynthesizerConfig{..} = cfg
    TaskFnDataset{..} = taskFnDataset
    variants :: [(String, Expr)] = (\(_k, v) -> (nodeRule v, v)) <$> exprBlocks
    in (:)
        (do
            -- let dataset = pickDataset datasets evaluateSet
            case synthesizer of
                "random" -> do
                    model <- A.sample RandomSynthesizerSpec
                    void $ train @device @rules @'[] @0 @RandomSynthesizer cfg taskFnDataset model
                "nsps" -> do
                    model :: NSPS device m symbols rules maxStringLength EncoderBatch R3nnBatch encoderChars typeEncoderChars h featMult
                            <- A.sample $ nspsSpec taskFnDataset variants r3nnBatch dropoutRate
                    model' <- if null savedModelPath
                        then pure model
                        else A.replaceParameters model . fmap D.IndependentTensor <$> D.load savedModelPath
                    void $ train @device @rules @'[R3nnBatch, maxStringLength * (2 * featMult * Dirs * h)] @(maxStringLength * m) cfg taskFnDataset model'
                    where
                    variants :: [(String, Expr)] = (\(_k, v) -> (nodeRule v, v)) <$> exprBlocks
                _ -> error "synthesizer not recognized"
            )
        $ getM @device @featMult @(m + 1) @rules @encoderChars @typeEncoderChars @symbols @maxStringLength @h cfg taskFnDataset
