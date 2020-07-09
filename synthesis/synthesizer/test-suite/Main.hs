{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

import           Test.Tasty                   (TestTree, defaultMain, testGroup)
import           Test.Tasty.Hspec
import           Test.Tasty.HUnit             ((@?=))

import           Spec.NSPS
import           Spec.Utility
import           Spec.Optimization

main ∷ IO ()
main = do
    synth_util_ <- testSpec "Synthesizer: Utility" synth_util
    nsps_ <- testSpec "NSPS" nsps
    optim_ <- testSpec "optim" optim
    let tree :: TestTree = testGroup "synthesis" [synth_util_, nsps_, optim_]
    defaultMain tree
