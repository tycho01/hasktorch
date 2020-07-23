{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

-- | generator logic
module Synthesis.Generator (module Synthesis.Generator) where

import System.Log.Logger
import System.ProgressBar
import System.Directory (removeFile, doesFileExist)
import Control.Exception (finally)
import Control.Monad (join, filterM, forM_, void, when)
import Data.Foldable (foldr')
import Data.Text (Text)
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
import Data.Set (Set)
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
    let fldr = resultFolder
    updateGlobalLogger logger . setLevel $ logPriority verbosity
    let gen :: StdGen = mkStdGen seed
    setStdGen gen
    let split = (training, validation, test)
    say_ "\ntypesByArity:"
    notice_ $ pp_ typesByArity
    say_ "\ndsl:"
    notice_ $ pp_ blockAsts

    say_ "\ngenerating task functions:"
    block_fn_types :: HashMap String Tp <- interpretUnsafe $ trackMapMHm fldr "block_fn_types" blockAsts exprType
    say_ "\nblock_fn_types:"
    notice_ $ pp_ block_fn_types
    -- Hint-based `exprType` approach fails here for exprs with type-ambiguous variables e.g. `show undefined`, even if type-annotated. typing this does work in GHCI so I should probably swap Hint out for the compiler API...
    unaliasedBlockTypes :: HashMap String Tp <- fromList <.> trackMapMList fldr "unaliasedBlockTypes" (toList block_fn_types) $ pure . first (pp . safeIndexHM blockAsts)
    say_ "\nunaliasedBlockTypes:"
    notice_ $ pp_ unaliasedBlockTypes
    let unaliasedVariants :: [(String, Expr)] = genBlockVariants unaliasedBlockTypes
    say_ "\nunaliasedVariants:"
    notice_ $ pp_ unaliasedVariants
    variantTypes :: [Tp] <- (tryType . snd) `mapM` unaliasedVariants
    -- variantTypes :: [Tp] <- trackMapMList fldr "variantTypes" unaliasedVariants $ tryType . snd
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
    fn_types :: HashMap Expr Tp <- interpretUnsafe $ trackFromKeysM fldr "fn_types" exprType task_fns
    let fn_types' :: HashMap String Tp = mapKeys pp fn_types
    say_ "\nfn_types:"
    notice_ $ pp_ fn_types

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
    task_instantiations :: [[Tp]] <- interpretUnsafe $ trackMapMList fldr "task_instantiations" task_types $ instantiateTypes typesByArity_ fill_types
    say_ "\ntask_instantiations:"
    notice_ $ pp_ task_instantiations
    -- for each function type, a list of type instantiations
    let type_fn_instantiations :: HashMap Tp [Tp] = fromList $ zip task_types task_instantiations
    say_ "\ntype_fn_instantiations:"
    notice_ $ pp_ type_fn_instantiations
    let type_instantiations :: HashMap String [([Tp], Tp)] = mapKeys pp $ fmap fnTypeIO <$> type_fn_instantiations
    say_ "\ntype_instantiations:"
    notice_ $ pp_ type_instantiations
    let in_type_instantiations :: [Tp] = nubPp . concat . fmap fst . concat . elems $ type_instantiations
    say_ "\nin_type_instantiations:"
    notice_ $ pp_ in_type_instantiations
    -- for each task function, for each type instantiation, the instantiated i/o types
    let fn_type_instantiations' :: HashMap Expr [([Tp], Tp)] = safeIndexHM type_instantiations . pp <$> fn_types
    let (fn_type_instantiations, gen'') :: (HashMap Expr [([Tp], Tp)], StdGen) =
            sampleHmLists gen maxInstantiations fn_type_instantiations'
    say_ "\nfn_type_instantiations:"
    notice_ $ pp_ fn_type_instantiations
    -- do sample generation not for each function but for each function input type
    -- for each non-function parameter combo type instantiation, a list of sample expressions
    rest_instantiation_inputs :: HashMap Tp [Expr] <- trackFromKeysM fldr "rest_instantiation_inputs" (pure . genInputs gen'' (numMin, numMax) (charMin, charMax) (listMin, listMax) numInputs) rest_type_instantiations
    say_ "\nrest_instantiation_inputs:"
    notice_ $ pp_ rest_instantiation_inputs
    -- map each instantiated function parameter type combination to a filtered map of generated programs matching its type
    let functionMatches :: Tp -> Expr -> Interpreter Bool = \ fn_type program_ast -> matchesType (safeIndexHM fn_types' $ pp program_ast) fn_type
    let filterFns :: Tp -> Interpreter [Expr] = \fn_type -> filterM (functionMatches fn_type) task_fns
    instantiated_fn_options :: HashMap Tp [Expr] <- interpretUnsafe $ trackFromKeysM fldr "instantiated_fn_options" filterFns in_type_instantiations
    say_ "\ninstantiated_fn_options:"
    notice_ $ pp_ instantiated_fn_options
    -- for each parameter combo type instantiation, a list of sample expressions
    let both_instantiation_inputs :: HashMap Tp [Expr] = rest_instantiation_inputs `union` instantiated_fn_options
    say_ "\nboth_instantiation_inputs:"

    -- hard-coding R3nnBatch as I can't pass it thru as config given the R3NN's LSTMs require it to be static
    let samplesPerInstantiation :: Int = fromIntegral . natVal $ Proxy @R3nnBatch
    fn_type_ios' :: HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)]) <- fromList <.> trackStatefulMapMList fldr "fn_type_ios'" gen'' (toList fn_type_instantiations) $ \ gen_ (fn, tp_instantiations) -> do
        -- notice_ $ "\nloop: " <> show (pp fn, bimap (fmap pp) pp <$> tp_instantiations)
        target_tp_io_pairs :: HashMap (Tp, Tp) [(Expr, Either String Expr)] <-
                interpretUnsafe $ fnOutputs crashOnError maxParams both_instantiation_inputs fn tp_instantiations
        -- some fns' tp instances get many i/o samples so let's nip that in the bud, cutting down to just the number our R3NN can take (i.e. we'd always train an instance on those couple samples)
        let (target_tp_io_pairs', gen') :: (HashMap (Tp, Tp) [(Expr, Either String Expr)], StdGen) =
                sampleHmLists gen_ samplesPerInstantiation target_tp_io_pairs
        return ((fn, target_tp_io_pairs'), gen')
    say_ "\nfn_type_ios':"
    notice_ $ pp_ fn_type_ios'
    say_ "filtering out program type instances without i/o samples"
    let fn_type_ios :: HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)]) = HM.filter (not . null) <$> fn_type_ios'
    say_ "\nfn_type_ios:"
    notice_ $ pp_ fn_type_ios
    say_ "combining i/o lists across type instances"
    let task_io_pairs :: HashMap Expr [(Expr, Either String Expr)] =
            join . elems <$> fn_type_ios
    say_ "\ntask_io_pairs:"
    notice_ $ pp_ task_io_pairs
    task_io_pairs' :: HashMap String [(Expr, Either String Expr)] <- fromList <.> trackMapMList fldr "task_io_pairs'" (toList task_io_pairs) $ pure . first pp
    say_ "\ntask_io_pairs':"
    notice_ $ pp_ task_io_pairs'

    say_ "checking sets contain no fns w/ behavior identical to any in other sets to prevent cheating"
    let kept_fns' :: [(Expr, (Tp, Tp))] = dedupeFunctions $ fn_type_ios
    say_ "\nkept_fns':"
    notice_ $ pp_ kept_fns'
    let kept_fn_strs :: Set String = Set.fromList $ pp_ <$> kept_fns'
    say_ "\nkept_fn_strs:"
    notice_ $ pp_ kept_fn_strs
    say_ "filtering to only used data"
    let fn_type_ios_ :: HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)]) = fmap fromList . pairs2lists . filter ((`Set.member` kept_fn_strs) . pp_ . second fst) . lists2pairs . fmap toList $ fn_type_ios
    say_ "\nfn_type_ios_:"
    notice_ $ pp_ fn_type_ios_
    say_ "taking type instantiations of task fns as separate entries"
    let tp_instances :: [(Expr, (Tp, Tp))] = lists2pairs $ keys <$> fn_type_ios_
    say_ "\ntp_instances:"
    notice_ $ pp_ tp_instances
    say_ "sampling task function type instances from any remaining programs"
    let kept_instances :: [(Expr, (Tp, Tp))] = take maxDataset . fst . fisherYates gen $ tp_instances
    say_ "\nkept_instances:"
    notice_ $ pp_ kept_instances
    let kept_fns :: [Expr] = keys $ pairs2lists kept_instances
    say_ "\nkept_fns:"
    notice_ $ pp_ kept_fns
    let fn_types_ = pickKeysByPp kept_fns fn_types

    say_ "calculating longest strings and character maps"
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
    let taskPath :: String = fldr <> "/" <> taskFile

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

    -- clear out tmp files for this run
    let tmpFiles = ["block_fn_types", "unaliasedBlockTypes", "variantTypes", "fn_types", "task_instantiations", "rest_instantiation_inputs", "instantiated_fn_options", "fn_type_ios", "task_io_pairs"]
    forM_ tmpFiles $ \name -> do
        let fpath :: String = fldr <> "/" <> name <> ".jsonl"
        exists :: Bool <- doesFileExist fpath
        when exists $ removeFile fpath
