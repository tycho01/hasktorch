{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

import           Test.Tasty                   (TestTree, defaultMain, testGroup)
import           Test.Tasty.Hspec
import           Test.Tasty.HUnit             ((@?=))

import           Prelude                      hiding (abs, all)
import           Control.Exception            (SomeException, try, evaluate)
import           Data.Int                     (Int64)
import           Data.Maybe                   (isNothing)
import           Data.Either                  (fromRight, isRight)
import           Data.Functor                 (void, (<&>))
import           Data.Bifunctor               (first, second)
import           Data.HashMap.Lazy            (HashMap, empty, insert, singleton, (!), keys, fromList)
import qualified Data.Set
import           System.Random                (StdGen, mkStdGen)
import           System.Timeout               (timeout)
import           Language.Haskell.Interpreter (as, interpret, liftIO, typeChecks, typeChecksWithDetails)
import           Util                         (fstOf3)
import           Language.Haskell.Interpreter

import           GHC.TypeNats
import           Torch.Typed.Functional
import qualified Torch.Tensor                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.Optim                   as D
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
import           Synthesis.Synthesizer.Utility
import           Synthesis.Synthesizer.Encoder
import           Synthesis.Synthesizer.R3NN
import           Synthesis.Synthesizer.NSPS

import           Synthesis.Synthesizer.Params

import           Spec.Ast
import           Spec.FindHoles
import           Spec.Generation
import           Spec.Hint
import           Spec.TypeGen
import           Spec.Types
import           Spec.Utility
import           Spec.Synthesizer.NSPS
import           Spec.Synthesizer.Synthesizer
import           Spec.Synthesizer.Utility
import           Spec.Synthesizer.Optimization

main ∷ IO ()
main = do
    util_ <- testSpec "Utility" util
    hint_ <- testSpec "Hint" hint
    gen_ <- testSpec "Generation" gen
    types_ <- testSpec "Types" types
    typeGen_ <- testSpec "TypeGen" typeGen
    find_ <- testSpec "FindHoles" find
    ast_ <- testSpec "Ast" ast
    synth_util_ <- testSpec "Synthesizer: Utility" synth_util
    nsps_ <- testSpec "NSPS" nsps
    optim_ <- testSpec "optim" optim
    synth_ <- testSpec "synth" synth
    let tree :: TestTree = testGroup "synthesis" [util_, types_, typeGen_, find_, ast_, hint_, gen_, synth_util_, nsps_, optim_, synth_]
    defaultMain tree
