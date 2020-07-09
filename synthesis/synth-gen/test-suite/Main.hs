{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

import           Test.Tasty                   (TestTree, defaultMain, testGroup)
import           Test.Tasty.Hspec
import           Test.Tasty.HUnit             ((@?=))

import           Spec.Ast
import           Spec.FindHoles
import           Spec.Generation
import           Spec.Hint
import           Spec.TypeGen
import           Spec.Types
import           Spec.Utility

main ∷ IO ()
main = do
    util_ <- testSpec "Utility" util
    hint_ <- testSpec "Hint" hint
    gen_ <- testSpec "Generation" gen
    types_ <- testSpec "Types" types
    typeGen_ <- testSpec "TypeGen" typeGen
    find_ <- testSpec "FindHoles" find
    ast_ <- testSpec "Ast" ast
    let tree :: TestTree = testGroup "synthesis" [util_, types_, typeGen_, find_, ast_, hint_, gen_]
    defaultMain tree
