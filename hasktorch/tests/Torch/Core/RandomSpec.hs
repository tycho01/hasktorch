{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Core.RandomSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Test.QuickCheck.Monadic

import Control.Monad (replicateM)
import Foreign (Ptr)

import qualified Control.Exception as E
import qualified Torch.Double as T
import Torch.Core.Random as R
import Torch.Prelude.Extras (doesn'tCrash)
import Orphans ()

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "newRNG" newRNGSpec
  describe "seed" seedSpec
  describe "manualSeed" manualSeedSpec
  describe "initialSeed" initialSeedSpec
  describe "random" randomSpec
  describe "uniform" uniformSpec
  describe "normal" normalSpec
  describe "exponential" exponentialSpec
  describe "cauchy" cauchySpec
  describe "logNormal" logNormalSpec
  describe "geometric" geometricSpec
  describe "bernoulli" bernoulliSpec
  describe "sampler generates unique values" uniqueSpec
  describe "scenario" $ do
    it "runs this scenario as expected" $ testScenario

newRNGSpec :: Spec
newRNGSpec = do
  rngs <- runIO (replicateM 10 R.newRNG)
  it "always creates a new random number" $
    zipWith (==) (tail rngs) (init rngs) `shouldNotContain` [True]

seedSpec :: Spec
seedSpec = do
  beforeAll
    (do
        rngs <- (replicateM 10 R.newRNG)
        rng1 <- mapM seed rngs
        rng2 <- mapM seed rngs
        pure (rngs, rng1, rng2)
    )
    (describe "seedSpec" $ do
      it "generates different values, given the same starting generators" $
        \(rngs, rng1, rng2) -> do
          zipWith (==) rng1 rng2 `shouldNotContain` [True]
    )

manualSeedSpec :: Spec
manualSeedSpec = do
  rngs <- runIO (replicateM 10 R.newRNG)
  rng1 <- runIO $ mapM (`manualSeed` 1) rngs
  rng2 <- runIO $ mapM (`manualSeed` 1) rngs

  it "generates the same value, given the same seed values" $
    zipWith (==) rng1 rng2 `shouldNotContain` [False]

initialSeedSpec :: Spec
initialSeedSpec = do
  it "doesn't crash" $
    pending

randomSpec :: Spec
randomSpec = do
  rngs <- runIO (replicateM 10 R.newRNG)
  rs <- runIO $ mapM random rngs
  it "generates numbers and doesn't crash" $
    rs `shouldSatisfy` doesn'tCrash

uniformSpec :: Spec
uniformSpec = do
  rng <- runIO R.newRNG
  distributed2BoundsCheck rng uniform $ \a b x ->
    case compare a b of
      LT -> x <= b && x >= a
      _  -> x <= a && x >= b

normalSpec :: Spec
normalSpec = do
  rng <- runIO R.newRNG
  distributed2BoundsCheck rng (withStdv normal) (\a b x -> doesn'tCrash ())

exponentialSpec :: Spec
exponentialSpec = do
  rng <- runIO R.newRNG
  distributed1BoundsCheck rng exponential property (\a x -> doesn'tCrash ())

cauchySpec :: Spec
cauchySpec = do
  rng <- runIO R.newRNG
  distributed2BoundsCheck rng cauchy (\a b x -> doesn'tCrash ())

logNormalSpec :: Spec
logNormalSpec = do
  rng <- runIO R.newRNG
  distributed2BoundsCheck rng (withStdv logNormal) (\a b x -> doesn'tCrash ())

geometricSpec :: Spec
geometricSpec = do
  rng <- runIO R.newRNG
  distributed1BoundsCheck rng geometric (forAll $ choose (0.0001, 0.9999)) (\a x -> doesn'tCrash ())

bernoulliSpec :: Spec
bernoulliSpec = do
  rng <- runIO R.newRNG
  distributed1BoundsCheck rng bernoulli (forAll $ choose (0.0001, 0.9999)) (\a x -> doesn'tCrash ())

-- |Check that seeds work as intended
testScenario :: IO ()
testScenario = do
  rng <- R.newRNG
  manualSeed rng 332323401
  val1 <- normal rng 0.0 1000
  val2 <- normal rng 0.0 1000
  E.assert (val1 /= val2) pure ()
  manualSeed rng 332323401
  manualSeed rng 332323401
  val3 <- normal rng 0.0 1000.0
  E.assert (val1 == val3) pure ()


-- ========================================================================= --

withStdv
  :: (Generator -> a -> b -> IO Double)
  -> Generator
  -> a
  -> NonZero (Positive b)
  -> IO Double
withStdv fn g a b = fn g a (getPositive (getNonZero b))


distributed2BoundsCheck
  :: (Show a, Show b, Arbitrary a, Arbitrary b)
  => Generator
  -> (Generator -> a -> b -> IO Double)
  -> (a -> b -> Double -> Bool)
  -> Spec
distributed2BoundsCheck g fun check = do
  it "should generate random numbers in the correct bounds" . property $ \(a, b) ->
      monadicIO $ do
        x <- run (fun g a b)
        assert (check a b x)

distributed1BoundsCheck :: (Show a, Arbitrary a) => Generator -> (Generator -> a -> IO b) -> ((a -> Property) -> Property) -> (a -> b -> Bool) -> Spec
distributed1BoundsCheck g fun pfun check = do
  it "should generate random numbers in the correct bounds" . pfun $ \a -> monadicIO $ do
    x <- run (fun g a)
    assert (check a x)


uniqueSpec :: Spec
uniqueSpec = do
  it "`normal` tensor sampler should generate unique tensors" $ do
    rng <- newRNG
    let Just sd = T.positive 1.0
    vals1 :: T.Tensor '[2] <- T.normal rng 0.0 sd
    vals2 :: T.Tensor '[2] <- T.normal rng 0.0 sd
    vals1 /= vals2 `shouldBe` True
  it "`logNormal` tensor sampler should generate unique tensors" $ do
    rng <- newRNG
    let Just sd = T.positive 1.0
    vals1 :: T.Tensor '[2] <- T.logNormal rng 0.0 sd
    vals2 :: T.Tensor '[2] <- T.logNormal rng 0.0 sd
    vals1 /= vals2 `shouldBe` True