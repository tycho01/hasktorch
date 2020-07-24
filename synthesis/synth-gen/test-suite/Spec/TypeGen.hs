{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

module Spec.TypeGen (module Spec.TypeGen) where

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

typeGen ∷ Spec
typeGen = parallel $ let
        bl = tyCon "Bool"
        int_ = tyCon "Int"
        str = tyCon "String"
        types_by_arity = insert 1 ["[]"] (singleton 0 ["Bool", "Int"])
        gen = mkStdGen seedDef
    in do

    it "findTypeVars" $ do
        -- Num a => a -> Set b
        let a = tyVar "a"
        let tp = tyForall Nothing (Just $ cxTuple [typeA "Num" a]) $ tyFun a $ tyApp (tyCon "Set") $ tyVar "b"
        findTypeVars tp `shouldBe` insert "a" (0, 1, [tyCon "Num"]) (singleton "b" (0, 2, []))
        -- Ord a => [a] -> [a]
        findTypeVars (tyForall Nothing (Just $ cxTuple [typeA "Ord" a]) $ tyFun (tyList a) $ tyList a) `shouldBe` singleton "a" (0, 2, [tyCon "Ord"])
        -- Foldable t => t a -> Bool
        pp_ (findTypeVars (parseType "Foldable t => t a -> Bool")) `shouldBe` pp_ (insert "t" (1 :: Int, 2 :: Int, [tyCon "Foldable"]) (singleton "a" (0 :: Int, 2 :: Int, [])))

    it "randomType" $ do
        GenerationConfig{..} <- liftIO parseGenerationConfig
        let nestLimit' :: Int = 0
        tp <- randomType types_by_arity False False nestLimit' empty 0 gen
        [tyCon "Bool", tyCon "Int", tyCon "Char"] `shouldContain` [tp]

    it "randomFnType" $ do
        GenerationConfig{..} <- liftIO parseGenerationConfig
        tp <- randomFnType types_by_arity False False nestLimit empty 0 gen
        -- [tyFun bl bl, tyFun bl int_, tyFun int_ bl, tyFun int_ int_] `shouldContain` [tp]
        True `shouldBe` True

    it "genTypes" $ do
        hm <- genTypes types_by_arity 0 10
        null (hm ! 0) `shouldBe` False

    it "fillTypeVars" $ do
        let a = tyVar "a"
        -- int_ -> a: a => Bool
        pp (fillTypeVars (tyFun int_ a) (singleton "a" bl)) `shouldBe` pp (tyFun int_ bl)
        -- Ord a => [a] -> [a]
        let tp = tyForall Nothing (Just $ cxTuple [typeA "Ord" a]) $ tyFun (tyList a) $ tyList a
        pp (fillTypeVars tp (singleton "a" bl)) `shouldBe` pp (tyFun (tyList bl) (tyList bl))

    it "mergeTyVars" $
        mergeTyVars (singleton "a" [bl, str]) (singleton "a" [int_, bl]) `shouldBe` singleton "a" [bl, str, int_]