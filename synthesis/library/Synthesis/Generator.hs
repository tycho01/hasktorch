{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

-- | generator logic
module Synthesis.Generator (module Synthesis.Generator) where

import System.Log.Logger
import Control.Monad (join, filterM, forM_)
import Data.Bifunctor (first)
import Data.HashMap.Lazy
  ( (!),
    HashMap,
    elems,
    filterWithKey,
    fromList,
    keys,
    mapWithKey,
    union,
    size,
  )
import Data.List (partition, maximum)
import qualified Data.Set as Set
import Data.Yaml
import System.Random (StdGen, mkStdGen)
import Language.Haskell.Interpreter (Interpreter)
import Synthesis.Blocks
import Synthesis.Generation
import Synthesis.Hint
import Synthesis.Ast
import Synthesis.Orphanage ()
import Synthesis.Types
import Synthesis.TypeGen
import Synthesis.Data (Expr, Tp, TaskFnDataset (..), GenerationConfig (..))
import Synthesis.Configs
import Synthesis.Utility
import Util (secondM)

-- | generate dataset
main :: IO ()
main = do
    cfg :: GenerationConfig <- parseGenerationConfig
    say_ $ show cfg
    let GenerationConfig {..} = cfg
    updateGlobalLogger logger . setLevel $ logPriority verbosity
    let gen :: StdGen = mkStdGen seed
    let split = (training, validation, test)
    say_ "\ntypesByArity:"
    notice_ $ pp_ typesByArity
    say_ "\ndsl:"
    notice_ $ pp_ blockAsts

    say_ "\ngenerating task functions:"
    block_fn_types :: HashMap String Tp <- interpretUnsafe $ mapM exprType blockAsts
    let variants :: [(String, Expr)] = genBlockVariants block_fn_types
    let dsl :: HashMap String Expr = filterWithKey (\k v -> k /= pp v) blockAsts
    programs :: [Expr] <- interpretUnsafe $ genFns maxHoles variants dsl
    say_ "\nprograms:"
    notice_ $ pp_ programs
    let task_fns = programs
    fn_types :: HashMap Expr Tp <- interpretUnsafe $ fromKeysM exprType task_fns
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
    let fns_rest :: ([Tp], [Tp]) = partition isFn input_types
    let mapRest :: [Tp] -> Interpreter [Tp] = concat <.> mapM (instantiateTypes typesByArity fill_types)
    (param_fn_types, rest_type_instantiations) :: ([Tp], [Tp]) <- interpretUnsafe $ secondM (nubPp <.> mapRest) $ first nubPp fns_rest
    say_ "\nparam_fn_types:"
    notice_ $ pp_ param_fn_types
    say_ "\nrest_type_instantiations:"
    notice_ $ pp_ rest_type_instantiations
    task_instantiations :: [[Tp]] <- interpretUnsafe $ instantiateTypes typesByArity fill_types `mapM` task_types
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
    -- for each function, for each type instantiation, for each param, the input type as string
    let fn_type_instantiations :: HashMap Expr [([Tp], Tp)] = (type_instantiations !) <$> fn_types
    say_ "\nfn_type_instantiations:"
    notice_ $ pp_ fn_type_instantiations
    -- do sample generation not for each function but for each function input type
    -- for each non-function parameter combo type instantiation, a list of sample expressions
    let rest_instantiation_inputs :: HashMap Tp [Expr] = fromKeys (genInputs gen (numMin, numMax) (charMin, charMax) (listMin, listMax) numInputs) rest_type_instantiations
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
    fn_type_ios :: HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)]) <- interpretUnsafe . sequence $ mapWithKey (fnOutputs crashOnError both_instantiation_inputs) fn_type_instantiations
    say_ "\nfn_type_ios:"
    notice_ $ pp_ fn_type_ios
    -- combine i/o lists across type instances
    let task_io_pairs :: HashMap Expr [(Expr, Either String Expr)] =
            join . elems <$> fn_type_ios

    -- filter out programs without i/o samples
    let kept_fns_ :: [Expr] = (not . null . (task_io_pairs !)) `filter` task_fns
    say_ "\nkept_fns_:"
    notice_ $ pp_ kept_fns_
    -- ensure sets contain no fns w/ behavior identical to any in other sets to prevent cheating
    let kept_fns' :: [Expr] = dedupeFunctions kept_fns_ fn_type_ios
    say_ "\nkept_fns':"
    notice_ $ pp_ kept_fns'
    -- sample task functions from any remaining programs
    let (shuffled, _gen) = fisherYates gen $ kept_fns'
    let kept_fns = take maxDataset shuffled
    say_ "\nkept_fns:"
    notice_ $ pp_ kept_fns
    let fn_types_ = pickKeys kept_fns fn_types
    let fn_type_ios_ = pickKeys kept_fns fn_type_ios

    let tp_pairs :: [(Tp, Tp)] = join . elems $ keys <$> fn_type_ios
    let longest_tp_string :: Int =
            maximum $ length <$> fmap (pp . fst) tp_pairs <> fmap (pp . snd) tp_pairs
    let ios :: [(Expr, Either String Expr)] =
            join . elems $ join . elems <$> fn_type_ios_

    let longest_expr_string :: Int =
            maximum $ length <$> fmap (pp . fst) ios <> fmap (pp_ . snd) ios
    let exprCharMap :: HashMap Char Int =
            indexChars $ (\(i,o) -> pp i <> pp_ o) <$> (join . join $ elems <$> elems fn_type_ios_)

    let longest_string :: Int = max longest_expr_string longest_tp_string
    let bothCharMap :: HashMap Char Int = mkCharMap $ elems fn_type_ios_
    -- TODO: this Hint-based `exprType` approach fails here for exprs with type-ambiguous variables e.g. `show undefined`, even if type-annotated. typing this does work in GHCI so I should probably swap Hint out for the compiler API...
    -- variantTypes :: [Tp] <- (exprType . letIn dsl . snd) `mapM` variants
    -- variantTypes :: [Tp] <- interpretUnsafe $ (tryType . letIn dsl . snd) `mapM` variants
    variantTypes :: [Tp] <- (tryType . letIn dsl . snd) `mapM` variants
    let ruleCharMap :: HashMap Char Int = indexChars $ pp <$> variantTypes
    let datasets :: ([Expr], [Expr], [Expr]) = randomSplit gen split kept_fns

    -- save task function data
    encodeFile taskPath $ TaskFnDataset
        cfg
        blockAsts
        typesByArity
        fn_types_
        fn_type_ios_
        rest_instantiation_inputs
        datasets
        variants
        variantTypes
        longest_expr_string
        longest_string
        exprCharMap
        bothCharMap
        ruleCharMap

    -- say_ "\n\nenumerating function i/o examples:"
    -- forM_ kept_fns $ \ast -> do
    --     let fn_type :: Tp = fn_types_ ! ast
    --     say_ "================================================"
    --     say_ $ "\n" ++ pp_ (expTypeSig (letRes ast) fn_type)
    --     let in_type_instance_outputs :: HashMap (Tp, Tp) [(Expr, Either String Expr)] = fn_type_ios_ ! ast
    --     say_ $ pp_ in_type_instance_outputs

    let set_list = untuple3 datasets
    forM_ (zip ["train", "validation", "test"] set_list) $ \(k, dataset) -> do
        say_ $ k <> ": " <> show (length dataset)
    let numSymbols = 1 + size blockAsts
    say_ $ "symbols: " <> show numSymbols
    let numRules = length variants
    say_ $ "rules: " <> show numRules
    say_ $ "max input/output string length: " <> show longest_string
    say_ $ "data written to " <> taskPath
