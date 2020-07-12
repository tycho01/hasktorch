{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

module Spec.Ast (module Spec.Ast) where

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

ast ∷ Spec
ast = parallel $ let
        bl = tyCon "Bool"
        int_ = tyCon "Int"
    in do

    it "skeleton" $ do
        pp (skeleton bl) `shouldBe` "undefined :: Bool"

    it "numAstNodes" $
        numAstNodes holeExpr `shouldBe` 3

    it "hasHoles" $ do
        hasHoles holeExpr `shouldBe` False
        let expr = expTypeSig holeExpr int_
        hasHoles expr `shouldBe` True

    it "genUncurry" $
        pp (genUncurry 2) `shouldBe` "\\ fn (a, b) -> fn a b"

    it "genInputs" $ do
        GenerationConfig{..} <- liftIO parseGenerationConfig
        let stdGen :: StdGen = mkStdGen seed
        let intRange = (numMin, numMax)
        let charRange = (charMin, charMax)
        let listLengths = (listMin, listMax)
        -- Bool
        pp <$> (genInputs stdGen intRange charRange listLengths 10 bl) `shouldContain` ["True"]
        -- -- [Bool]
        -- let lists = genInputs stdGen intRange charRange listLengths 10 $ tyList bl
        -- (length . nubPp . concat . fmap unList) lists `shouldBe` 2

    it "genHoledVariants" $ do
        let tp = parseType "Int -> String -> Tp"
        fmap pp (genHoledVariants "f" tp) `shouldBe` ["f", "f (undefined :: Int)", "f (undefined :: Int) (undefined :: String)"]
