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
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-full-laziness #-}

module Synthesizer.Train (module Synthesizer.Train) where

import           System.Random                 (StdGen, mkStdGen, setStdGen)
import           System.Timeout                (timeout)
import           System.Directory              (createDirectoryIfMissing)
import           System.CPUTime
import           System.ProgressBar
import           Data.Text.Internal.Lazy (Text)
import           Data.Maybe                    (fromMaybe)
import           Data.Set                      (Set, empty, insert)
import qualified Data.Set
import           Data.Bifunctor                (first, second)
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Internal      as BS
import qualified Data.ByteString.Lazy.Internal as BL
import           Data.HashMap.Lazy             (HashMap, (!), elems, keys, size, mapWithKey, filterWithKey, fromListWith, singleton)
import qualified Data.Csv as Csv
import           Data.Text.Prettyprint.Doc (pretty)
import           Text.Printf
import           Foreign.Marshal.Utils         (fromBool)
import           Control.Monad                 (join, replicateM, forM, void, when)
import           Control.Monad.Trans.Loop
import           Control.Monad.Trans.Class
import           Language.Haskell.Exts.Syntax  ( Exp (..) )
import           Prelude                        hiding (abs)
import           Util                          (fstOf3)
import           Language.Haskell.Interpreter  ( Interpreter, liftIO, lift )
import           GHC.Exts
import           GHC.Generics                  (Generic)
import           GHC.TypeNats                  (KnownNat, Nat, CmpNat, type (*), type (-))
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
import           Synthesizer.R3NN
import           Synthesizer.Synthesizer
import           Synthesizer.Params

-- | pre-calculate DSL stuff
prep_dsl :: TaskFnDataset -> PreppedDSL
prep_dsl TaskFnDataset{..} =
    PreppedDSL variants variant_sizes dsl'
    where
    variants :: [(String, Expr)] = (\(_k, v) -> (nodeRule v, v)) <$> exprBlocks
    variant_sizes :: HashMap String Int = fromList $ variantInt . snd <$> variants
    dsl' = filterWithKey (\k v -> k /= pp v) dsl

-- | train a NSPS model and return results
train :: forall device rules shape synthesizer . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat rules, KnownShape shape, Synthesizer device shape rules synthesizer, KnownNat (FromMaybe 0 (ExtractDim BatchDim shape)), TensorOptions shape 'D.Float device) => SynthesizerConfig -> TaskFnDataset -> synthesizer -> Interpreter [EvalResult]
train synthesizerConfig taskFnDataset init_model = do
    let SynthesizerConfig{..} = synthesizerConfig
    let TaskFnDataset{..} = taskFnDataset
    let init_lr :: Tensor device 'D.Float '[] = UnsafeMkTensor . D.asTensor $ learningRate
    let prepped_dsl = prep_dsl taskFnDataset
    let PreppedDSL{..} = prepped_dsl
    let init_optim :: D.Adam = d_mkAdam 0 0.9 0.999 $ A.flattenParameters init_model

    _ <- iterateLoopT (1 :: Int) $ \ !epoch -> do
        lift $ notice $ "epoch: " <> show epoch
        let n :: Int = 1000
        pb <- lift . liftIO $ newProgressBar pgStyle 1 (Progress 0 n ("task-fns" :: Text))

        let model = init_model
        let dummy :: Tensor device 'D.Float '[] = zeros
        -- TRAIN LOOP
        (optim', _) :: (D.Adam, Int) <- lift $ iterateLoopT (init_optim, 0) $ \ !state@(optim, task_fn_id_) -> if task_fn_id_ >= n then exitWith state else do
                let loss :: Tensor device 'D.Float '[] = patchLoss @device @shape @rules model variant_sizes dummy
                (newParam, optim') <- lift . liftIO $ doStep @device @shape @rules model optim loss init_lr
                -- let optim' = optim
                lift . liftIO $ incProgress pb 1
                return (optim', task_fn_id_ + 1)

    return []
