{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

-- | generator logic
module Synthesis.Generator (module Synthesis.Generator) where

import System.Log.Logger
import System.ProgressBar
import System.Directory (doesFileExist, removeFile)
import Control.Exception (finally)
import Control.Monad (join, filterM, forM_, void)
import Data.Foldable (foldr', foldrM)
import Data.Text (Text)
import qualified Data.ByteString as BS
import Data.ByteString.Char8 (pack)
import Data.ByteString.Lazy (toStrict, fromStrict)
import Data.Bifunctor (first, second, bimap)
import Data.HashMap.Lazy
  ( (!),
    HashMap,
    elems,
    filterWithKey,
    fromList,
    toList,
    keys,
    mapWithKey,
    union,
    size,
    lookupDefault,
    fromListWith,
  )
import qualified Data.HashMap.Lazy as HM
import Data.List (partition, maximum)
import Data.Maybe (catMaybes, fromJust)
import qualified Data.Set as Set
import qualified Data.Aeson as Aeson
import Data.Yaml
import Data.Proxy
import System.Random (StdGen, mkStdGen, setStdGen)
import GHC.TypeNats (natVal)
import Language.Haskell.Interpreter (Interpreter)
import Synthesis.Blocks
import Synthesis.Generation
import Synthesis.Hint
import Synthesis.Ast
import Synthesis.Orphanage ()
import Synthesis.Types
import Synthesis.TypeGen
import Synthesis.Data
import Synthesis.Configs
import Synthesis.Utility
import Util (secondM)

-- | generate dataset
main :: IO ()
main = do
    cfg :: GenerationConfig <- parseGenerationConfig
    say_ $ show cfg
    encodeFile "./gencfg.yml" cfg
    let GenerationConfig {..} = cfg
    updateGlobalLogger logger . setLevel $ logPriority verbosity
    let gen :: StdGen = mkStdGen seed
    setStdGen gen
    let split = (training, validation, test)
    say_ "\ntypesByArity:"
    notice_ $ pp_ typesByArity
    say_ "\ndsl:"
    notice_ $ pp_ blockAsts

    say_ "\ngenerating task functions:"
    block_fn_types :: HashMap String Tp <- interpretUnsafe $ mapM exprType blockAsts
    say_ "\nblock_fn_types:"
    notice_ $ pp_ block_fn_types
    -- Hint-based `exprType` approach fails here for exprs with type-ambiguous variables e.g. `show undefined`, even if type-annotated. typing this does work in GHCI so I should probably swap Hint out for the compiler API...
    let unaliasedBlockTypes :: HashMap String Tp = asPairs (first (pp . (safeIndexHM blockAsts))) block_fn_types
    say_ "\nunaliasedBlockTypes:"
    notice_ $ pp_ unaliasedBlockTypes
    let unaliasedVariants :: [(String, Expr)] = genBlockVariants unaliasedBlockTypes
    say_ "\nunaliasedVariants:"
    notice_ $ pp_ unaliasedVariants
    variantTypes :: [Tp] <- (tryType . snd) `mapM` unaliasedVariants
    say_ "\nvariantTypes:"
    notice_ $ pp_ variantTypes
    -- characters used in types: manually add stuff that may be added in types of synthesized functions
    let ruleCharMap :: HashMap Char Int = indexChars $ (:) "1234567890\n" $ pp <$> variantTypes
    say_ "\nruleCharMap:"
    notice_ $ show ruleCharMap
    let variants :: [(String, Expr)] = genBlockVariants block_fn_types
    say_ "\nvariants:"
    notice_ $ pp_ variants
    let dsl :: HashMap String Expr = filterWithKey (\k v -> k /= pp v) blockAsts
    programs :: [Expr] <- interpretUnsafe $ genFns maxHoles variants dsl
    say_ "\nprograms:"
    notice_ $ pp_ programs
    let task_fns = programs
    fn_types :: HashMap Expr Tp <- interpretUnsafe $ fromKeysM exprType task_fns
    say_ "\nfn_types:"
    notice_ $ pp_ fn_types

    exists :: Bool <- doesFileExist jsonLinesPath
    if exists then pure () else do
        let task_types :: [Tp] = elems fn_types
        say_ "\ntask_types:"
        notice_ $ pp_ task_types
        -- generated types we will use for instantiating type variables
        fill_types :: HashMap Int [Tp] <- genTypes typesByArity nestLimit maxInstances
        say_ "\nfill_types:"
        notice_ $ pp_ fill_types
        let fn_input_types :: HashMap Expr [Tp] = fnInputTypes <$> fn_types
        say_ "\nfn_input_types:"
        notice_ $ pp_ fn_input_types
        let input_types :: [Tp] = nubPp . concat . elems $ fn_input_types
        say_ "\ninput_types:"
        notice_ $ pp_ input_types
        -- split the input types for our programs into functions vs other -- then instantiate others.
        let typesByArity_ :: HashMap Int [Tp] = fmap parseType <$> typesByArity
        let fns_rest :: ([Tp], [Tp]) = partition isFn input_types
        let mapRest :: [Tp] -> Interpreter [Tp] = concat <.> mapM (instantiateTypes typesByArity_ fill_types)
        (param_fn_types, rest_type_instantiations) :: ([Tp], [Tp]) <- interpretUnsafe $ secondM (nubPp <.> mapRest) $ first nubPp fns_rest
        say_ "\nparam_fn_types:"
        notice_ $ pp_ param_fn_types
        say_ "\nrest_type_instantiations:"
        notice_ $ pp_ rest_type_instantiations
        task_instantiations :: [[Tp]] <- interpretUnsafe $ instantiateTypes typesByArity_ fill_types `mapM` task_types
        say_ "\ntask_instantiations:"
        notice_ $ pp_ task_instantiations
        -- for each function type, a list of type instantiations
        let type_fn_instantiations :: HashMap Tp [Tp] = fromList $ zip task_types task_instantiations
        say_ "\ntype_fn_instantiations:"
        notice_ $ pp_ type_fn_instantiations
        let type_instantiations :: HashMap Tp [([Tp], Tp)] = fmap fnTypeIO <$> type_fn_instantiations
        say_ "\ntype_instantiations:"
        notice_ $ pp_ type_instantiations
        let in_type_instantiations :: [Tp] = nubPp . concat . fmap fst . concat . elems $ type_instantiations
        say_ "\nin_type_instantiations:"
        notice_ $ pp_ in_type_instantiations
        -- for each function, for each param, the instantiated i/o types
        let fn_type_instantiations' :: HashMap Expr [([Tp], Tp)] = (type_instantiations !) <$> fn_types
        let (fn_type_instantiations, gen'') :: (HashMap Expr [([Tp], Tp)], StdGen) =
                sampleHmLists gen maxInstantiations fn_type_instantiations'
        say_ "\nfn_type_instantiations:"
        notice_ $ pp_ fn_type_instantiations
        -- do sample generation not for each function but for each function input type
        -- for each non-function parameter combo type instantiation, a list of sample expressions
        let rest_instantiation_inputs :: HashMap Tp [Expr] = fromKeys (genInputs gen'' (numMin, numMax) (charMin, charMax) (listMin, listMax) numInputs) rest_type_instantiations
        say_ "\nrest_instantiation_inputs:"
        notice_ $ pp_ rest_instantiation_inputs
        -- map each instantiated function parameter type combination to a filtered map of generated programs matching its type
        let functionMatches :: Tp -> Expr -> Interpreter Bool = \fn_type program_ast -> matchesType (fn_types ! program_ast) fn_type
        let filterFns :: Tp -> Interpreter [Expr] = \fn_type -> filterM (functionMatches fn_type) task_fns
        instantiated_fn_options :: HashMap Tp [Expr] <- interpretUnsafe $ fromKeysM filterFns in_type_instantiations
        say_ "\ninstantiated_fn_options:"
        notice_ $ pp_ instantiated_fn_options
        -- for each parameter combo type instantiation, a list of sample expressions
        let both_instantiation_inputs :: HashMap Tp [Expr] = rest_instantiation_inputs `union` instantiated_fn_options
        say_ "\nboth_instantiation_inputs:"
        notice_ $ pp_ both_instantiation_inputs
        pb <- newProgressBar pgStyle 1 (Progress 0 (size fn_type_instantiations) "generator")
        writeFile jsonLinesPath ""
        -- hard-coding R3nnBatch as I can't pass it thru as config given the R3NN's LSTMs require it to be static
        let samplesPerInstantiation :: Int = fromIntegral . natVal $ Proxy @R3nnBatch
        let foldIOs = \ (fn, type_instantiations) gen_ -> do
                    notice_ $ "\nloop: " <> show (pp fn, bimap (fmap pp) pp <$> type_instantiations)
                    target_tp_io_pairs :: HashMap (Tp, Tp) [(Expr, Either String Expr)] <-
                            interpretUnsafe $ fnOutputs crashOnError maxParams both_instantiation_inputs fn type_instantiations
                    -- some fns' tp instances get many i/o samples so let's nip that in the bud, cutting down to just the number our R3NN can take (i.e. we'd always train an instance on those couple samples)
                    let (target_tp_io_pairs', gen') :: (HashMap (Tp, Tp) [(Expr, Either String Expr)], StdGen) =
                            sampleHmLists gen_ samplesPerInstantiation target_tp_io_pairs
                    BS.appendFile jsonLinesPath . (<> pack "\n") . toStrict . Aeson.encode . (fn,) $ target_tp_io_pairs'
                    incProgress pb 1
                    return gen'
        void $ foldrM foldIOs gen (toList fn_type_instantiations)

    fn_type_ios :: HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)]) <- fromList . fmap (fromJust . Aeson.decode . fromStrict . pack) . init . lines <$> readFile jsonLinesPath
    say_ "\nfn_type_ios:"
    notice_ $ pp_ fn_type_ios
    say_ "combining i/o lists across type instances"
    let task_io_pairs :: HashMap Expr [(Expr, Either String Expr)] =
            join . elems <$> fn_type_ios
    let task_io_pairs' :: HashMap String [(Expr, Either String Expr)] = mapKeys pp task_io_pairs

    say_ "filtering out programs without i/o samples"
    let kept_fns_ :: [Expr] = (not . null . ((\k -> lookupDefault [] k task_io_pairs') . pp)) `filter` task_fns
    say_ "\nkept_fns_:"
    notice_ $ pp_ kept_fns_
    say_ "checking sets contain no fns w/ behavior identical to any in other sets to prevent cheating"
    let kept_fns' :: [Expr] = dedupeFunctions kept_fns_ fn_type_ios
    say_ "\nkept_fns':"
    notice_ $ pp_ kept_fns'
    say_ "filtering out program type instances without i/o samples"
    let fn_type_ios_ = HM.filter (not . null) <$> pickKeysByPp kept_fns' fn_type_ios
    say_ "taking type instantiations of task fns as separate entries"
    let tp_instances :: [(Expr, (Tp, Tp))] = lists2pairs $ keys <$> fn_type_ios_
    say_ "sampling task function type instances from any remaining programs"
    let kept_instances :: [(Expr, (Tp, Tp))] = take maxDataset . fst . fisherYates gen $ tp_instances
    let kept_fns :: [Expr] = keys $ pairs2lists kept_instances
    say_ "\nkept_fns:"
    notice_ $ pp_ kept_fns
    let fn_types_ = pickKeysByPp kept_fns fn_types

    let tp_pairs :: [(Tp, Tp)] = join . elems $ keys <$> fn_type_ios_
    let longest_tp_string :: Int =
            maximum $ length <$> fmap (pp . fst) tp_pairs <> fmap (pp . snd) tp_pairs <> fmap pp variantTypes
    let ios :: [(Expr, Either String Expr)] =
            join . elems $ join . elems <$> fn_type_ios_

    let longest_expr_string :: Int =
            maximum $ length <$> fmap (pp . fst) ios <> fmap (pp_ . snd) ios
    let exprCharMap :: HashMap Char Int =
            indexChars $ (\(i,o) -> pp i <> pp_ o) <$> (join . join $ elems <$> elems fn_type_ios_)

    let longest_string :: Int = max longest_expr_string longest_tp_string
    let bothCharMap :: HashMap Char Int = mkCharMap $ elems fn_type_ios_
    let datasets :: Tpl3 (HashMap Expr [(Tp, Tp)]) = mapTuple3 pairs2lists . randomSplit gen split $ kept_instances

    -- save task function data
    encodeFile taskPath $ TaskFnDataset
        cfg
        blockAsts
        typesByArity
        fn_types_
        fn_type_ios_
        datasets
        variants
        variantTypes
        longest_expr_string
        longest_string
        exprCharMap
        bothCharMap
        ruleCharMap

    let set_list = untuple3 datasets
    forM_ (zip ["train", "validation", "test"] set_list) $ \(k, dataset) -> do
        say_ $ k <> ": " <> show (length dataset)
    let numSymbols = 1 + size blockAsts
    say_ $ "symbols: " <> show numSymbols
    let numRules = length variants
    say_ $ "rules: " <> show numRules
    say_ $ "max input/output string length: " <> show longest_string
    say_ $ "data written to " <> taskPath
    removeFile jsonLinesPath
