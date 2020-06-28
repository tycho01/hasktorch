{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Main where

import System.Process (system)
import Dataset
import Graphics.Vega.VegaLite hiding (sample, shape)
import Plot
import Torch
import Model

data (Parameterized m) => PruneSpec m = PruneSpec {
    ref :: m,
    selectWeights :: m -> [Tensor],
    pruneWeights :: Float -> Tensor -> Tensor
}

selectAllWeights model = 
    toDependent <$> flattenParameters model

l1Prune :: Float -> Tensor -> Tensor
l1Prune threshold t =
    Torch.abs t `lt` (asTensor threshold)

l1 :: Tensor -> Tensor
l1 w = l1Loss ReduceMean w (zerosLike w)

l2 :: Tensor -> Tensor
l2 = mean

-- combinedLoss :: CNN -> Tensor -> Tensor -> Tensor -- model, input, target
{-
combinedLoss model input target =
    let regWeights = flattenAll $ cat (Dim 0) $ flattenAll <$> (selectWeights pruneSpec $ initRef) 
        in 0 * (nllLoss' (forward model input) target) + 1000000.0 * l1 regWeights 
-}

-- | Setup pruning parameters and run
runPrune :: (Dataset d) => d -> IO (CNN, CNN) 
runPrune mnistData = do

    print "sampling"
    -- train reference model
    initRef <- sample refSpec
    let optimSpec = OptimSpec {
        -- optimizer = GD,
        optimizer = mkAdam 0 0.9 0.999 (flattenParameters initRef),
        batchSize = 128,
        numIters = 50,
        learningRate = 5e-5, 
        lossFn = \model input target -> nllLoss' target (forward model input)
    } :: OptimSpec Adam CNN

    print "training"
    ref <- train optimSpec mnistData =<< sample refSpec
    
    let pruneSpec = PruneSpec {
        ref = initRef,
        -- selectWeights = (:[]) . toDependent . weight . cnnFC0 
        selectWeights = \m -> [toDependent . weight . cnnFC0 $ m,
                               toDependent . weight . cnnFC1 $ m]
    }
            
    print "l1"
    let combinedL1 = \model input target ->
            let regWeights = flattenAll $ cat (Dim 0) $ flattenAll <$> (selectWeights pruneSpec $ initRef) 
                in 0 * (nllLoss' (forward model input) target) + 1000000.0 * l1 regWeights 
    l1Model <- train optimSpec { lossFn = combinedL1 } mnistData =<< sample refSpec

    print "l2"
    let combinedL2 = \model input target ->
            let regWeights = flattenAll $ cat (Dim 0) $ flattenAll <$> (selectWeights pruneSpec $ initRef) 
                in 0 * (nllLoss' (forward model input) target) + 1000000.0 * l2 regWeights 
    l2Model <- train optimSpec { lossFn = combinedL2} mnistData =<< sample refSpec

    print "weights0 init"

    plt <- strip (toDependent . weight . cnnFC0 $ initRef)
    toHtmlFile "plotInit.html" plt
    system "open plotInit.html"

    print "weights0 l1"
    plt <- strip (toDependent . weight . cnnFC0 $ l1Model)
    toHtmlFile "plot0l1.html" plt
    system "open plot0l1.html"

    print "weights0 l2"
    plt <- strip (toDependent . weight . cnnFC0 $ l2Model)
    toHtmlFile "plot0l2.html" plt
    system "open plot0l2.html"

    let resultWt = head $ (selectWeights pruneSpec) l1Model
    -- print $ resultWt
    print $ shape resultWt

    print "pruning"
    let pruned = undefined
    pure pruned
    -- pure (ref, pruned)
  where
    channels = 3
    refSpec = 
        CNNSpec
            -- input channels, output channels, kernel height, kernel width
            (Conv2dSpec 1 channels 5 5)
            -- (no ADT : maxPool2d (3, 3) (3, 3) (0, 0) (1, 1) Floor )
            (LinearSpec (9*9*channels) 80)
            (LinearSpec 80 10)

pruneTest = do
    putStrLn "Loading Data"
    (mnistTrain, mnistTest) <- loadMNIST "datasets/mnist"
    putStrLn "Running Prune"
    (original, derived) <- runPrune mnistTrain
    putStrLn "Done"

gradTest = do
    let foo = asTensor ([1, 2, 3] :: [Float])
    bar <- makeIndependent foo
    let baz = sumAll $ Torch.abs $ toDependent bar
    let baz' = sumAll $ pow (2 :: Int) $ toDependent bar
    print "l1"
    print $ grad baz [bar]
    print "l2"
    print $ grad baz' [bar]
    -- print "no"
    -- print $ grad (sumAll $ foo) [bar]
    -- print "no2"
    -- bar' <- makeIndependent foo
    -- print $ grad baz [bar, bar']
    pure ()

main = pruneTest