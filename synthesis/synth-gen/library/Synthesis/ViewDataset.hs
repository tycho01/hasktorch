{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE TypeOperators #-}

-- | view some info about a dataset
module Synthesis.ViewDataset (module Synthesis.ViewDataset) where

import Data.HashMap.Lazy
import Control.Monad (forM_)
import Data.Yaml
import Synthesis.Hint
import Synthesis.Orphanage ()
import Synthesis.Data
import Synthesis.Configs
import Synthesis.Utility
import Synthesis.Types
import Synthesis.Ast

-- | main function
main :: IO ()
main = do
    cfg :: ViewDatasetConfig <- parseViewDatasetConfig
    say_ $ show cfg
    let ViewDatasetConfig{..} = cfg
    taskFnDataset :: TaskFnDataset <- decodeFileThrow taskPath
    let TaskFnDataset{..} = taskFnDataset
    let (train_set, _validation_set, _test_set) = datasets
    say_ $ show generationCfg
    -- say_ $ show taskFnDataset
    let set_list = untuple3 datasets
    forM_ (zip ["train", "validation", "test"] set_list) $ \(k, dataset) -> do
        say_ $ k <> ": " <> show (length dataset)
    printTaskFns taskFnDataset train_set

-- | print info on task functions
printTaskFns :: TaskFnDataset -> HashMap Expr [(Tp, Tp)] -> IO ()
printTaskFns TaskFnDataset{..} train_set = do
    say_ "\n\nenumerating function i/o examples:"
    forM_ (keys train_set) $ \ast -> do
        let fn_type :: Tp = fnTypes ! ast
        say_ "================================================"
        say_ $ "\n" ++ pp_ (expTypeSig (letRes ast) fn_type)
        let tp_ios :: HashMap (Tp, Tp) [(Expr, Either String Expr)] = fnTypeIOs ! ast
        notice_ $ pp_ tp_ios
