{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

module Spec.Optimization (module Spec.Optimization) where

import           Test.Tasty                   (TestTree, defaultMain, testGroup)
import           Test.Tasty.Hspec
import           Test.Tasty.HUnit             ((@?=))

import           Prelude                      hiding (abs, all)
import           Control.Exception            (SomeException, try, evaluate)
import           Control.Monad (mapM, join)
import           Data.Int                     (Int64)
import           Data.Maybe                   (isNothing)
import           Data.Either                  (fromRight, isRight)
import           Data.Functor                 (void, (<&>))
import           Data.Bifunctor               (first, second)
import           Data.HashMap.Lazy            (HashMap, empty, insert, singleton, (!), keys, fromList, size)
import qualified Data.Set as Set
import           Data.Yaml
import           System.Random                (StdGen, mkStdGen)
import           System.Timeout               (timeout)
import           Language.Haskell.Interpreter (as, interpret, liftIO, typeChecks, typeChecksWithDetails)
import           Util                         (fstOf3)
import           Language.Haskell.Interpreter

import           GHC.TypeNats
import           Torch.Typed.Functional
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.DType                   as D
import qualified Torch.Optim                   as D
import qualified Torch.Autograd                as D
import qualified Torch.Functional.Internal     as I
import qualified Torch.Functional              as F
import qualified Torch.NN                      as A
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.Factories
import           Torch.Typed.Optim
import           Torch.Typed.Parameter
import           Torch.Typed.NN
import           Torch.Typed.NN.Recurrent.LSTM

import           Synthesis.Ast
import           Synthesis.Configs
import           Synthesis.Blocks
import           Synthesis.FindHoles
import           Synthesis.Generation
import           Synthesis.Hint
import           Synthesis.Types
import           Synthesis.TypeGen
import           Synthesis.Data
import           Synthesis.Utility
import           Synthesizer.Utility
import           Synthesizer.Encoder
import           Synthesizer.R3NN
import           Synthesizer.NSPS
import           Synthesizer.Train

import           Synthesizer.Params
import           Synthesizer.GridSearch
import           Spec.Types

type Device = Cpu

optim ∷ Spec
optim = parallel $ do

    it "static parameter combinations" $ do
        True `shouldBe` True  -- comment test as it presumes existing dataset file
        -- EvolutionaryConfig{..} <- parseEvolutionaryConfig
        -- let cfg = OptimizationConfig{..}
        -- taskFnDataset :: TaskFnDataset <- decodeFileThrow taskPath
        -- let TaskFnDataset{..} = taskFnDataset
        -- (length $ (flip (!!) $ head mOpts) $ getM @Device @FeatMultWithTypes @0 @0 @0 @0 @0 @0 cfg taskFnDataset hparCombs) `shouldBe` (length hparCombs `div` length mOpts)
        -- (length . join        $ pickIdxs mOpts $ getM @Device @FeatMultWithTypes @0 @0 @0 @0 @0 @0 cfg taskFnDataset hparCombs) `shouldBe` length hparCombs
        -- (length . join . join $ pickIdxs hOpts $ getH @Device @FeatMultWithTypes @0 @0 @0 @0 @0    cfg taskFnDataset hparCombs) `shouldBe` length hparCombs * length mOpts
        -- -- say_ . show $ length hparCombs * length mOpts * length hOpts
        -- -- (length . join . join $ (!! longestString) $ getMaxStringLength @Device @FeatMultWithTypes @0 @0 @0 @0 cfg taskFnDataset hparCombs) `shouldBe` length hparCombs * length mOpts * length hOpts
        -- -- (length . join . join $ (!! (size dsl + natValI @LhsSymbols)) $ getSymbols @Device @FeatMultWithTypes @0 @0 @0 cfg taskFnDataset hparCombs) `shouldBe` length hparCombs * length mOpts * length hOpts
        -- -- (length . join . join $ (!! (size charMap + 1)) $ getMaxChar @Device @FeatMultWithTypes @0 @0 cfg taskFnDataset hparCombs) `shouldBe` length hparCombs * length mOpts * length hOpts
        -- -- (length . join . join $ (!! length exprBlocks) $ getRules @Device @FeatMultWithTypes @0 cfg taskFnDataset hparCombs) `shouldBe` length hparCombs * length mOpts * length hOpts
