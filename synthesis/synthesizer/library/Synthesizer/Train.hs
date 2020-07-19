{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE Strict #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fno-full-laziness #-}

module Synthesizer.Train (module Synthesizer.Train) where

import           System.Random                 (StdGen, mkStdGen, setStdGen)
import           System.Timeout                (timeout)
import           System.Directory              (createDirectoryIfMissing)
import           System.CPUTime
import           System.ProgressBar
import           Data.Text.Internal.Lazy (Text)
import           Data.Maybe                    (fromMaybe)
import           Data.Set                      (Set, empty, insert)
import qualified Data.Set
import           Data.Bifunctor                (first, second)
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Internal      as BS
import qualified Data.ByteString.Lazy.Internal as BL
import           Data.HashMap.Lazy             (HashMap, (!), elems, keys, size, mapWithKey, filterWithKey, fromListWith, singleton)
import qualified Data.Csv as Csv
import           Data.Text.Prettyprint.Doc (pretty)
import           Text.Printf
import           Foreign.Marshal.Utils         (fromBool)
import           Control.Monad                 (join, replicateM, forM, void, when)
import           Control.Monad.Trans.Loop
import           Control.Monad.Trans.Class
import           Language.Haskell.Exts.Syntax  ( Exp (..) )
import           Prelude                        hiding (abs)
import           Util                          (fstOf3)
import           Language.Haskell.Interpreter  ( Interpreter, liftIO, lift )
import           GHC.Exts
import           GHC.Generics                  (Generic)
import           GHC.TypeNats                  (KnownNat, Nat, CmpNat, type (*), type (-))
import qualified Torch.Functional.Internal     as I
import qualified Torch.DType                   as D
import qualified Torch.Tensor                  as D
import qualified Torch.Device                  as D
import qualified Torch.TensorFactories         as D
import qualified Torch.TensorOptions           as D
import qualified Torch.Optim                   as D
import qualified Torch.Serialize               as D
import qualified Torch.Autograd                as D
import qualified Torch.Functional              as F
import qualified Torch.NN                      as A
import           Torch.Typed.NN.Recurrent.LSTM
import           Torch.Typed.Aux
import           Torch.Typed.Tensor
import           Torch.Typed.NN
import           Torch.Typed.Parameter
import qualified Torch.Typed.Parameter
import           Torch.Typed.Factories
import           Torch.Typed.Optim
import           Torch.Typed.Functional
import           Torch.Typed.Autograd
import           Torch.Typed.Serialize
import qualified Torch.Distributions.Distribution as Distribution
import qualified Torch.Distributions.Categorical as Categorical

import           Synthesis.Orphanage ()
import           Synthesis.Data hiding (GridSearchConfig(..), EvolutionaryConfig(..))
import           Synthesis.Utility
import           Synthesis.Ast
import           Synthesis.Generation
import           Synthesis.FindHoles
import           Synthesis.Hint
import           Synthesis.Types
import           Synthesizer.Utility
import           Synthesizer.Encoder
import           Synthesizer.R3NN
import           Synthesizer.Synthesizer
import           Synthesizer.Params

-- -- | deterministically pick the most likely expansion to fill a hole in a PPT
-- -- | deprecated, not in use
-- -- TODO: replace this whole block with argmaxAll (argmax_t) returning non-flat index (like np.unravel_index)...
-- -- (hole_idx, rule_idx) :: (Int, Int) = unravelIdx . argmaxAll $ hole_expansion_probs
-- argmaxExpansion :: forall num_holes rules device . Tensor device 'D.Float '[num_holes, rules] -> (Int, Int)
-- argmaxExpansion hole_expansion_probs = (hole_idx, rule_idx) where
--     (hole_dim, rule_dim) :: (Int, Int) = (0, 1)
--     rule_idx_by_hole :: Tensor device 'D.Int64 '[num_holes] =
--             asUntyped (F.argmax (F.Dim rule_dim) F.RemoveDim) hole_expansion_probs
--     num_holes_ :: Int = shape' rule_idx_by_hole !! 0
--     best_prob_by_hole :: Tensor device 'D.Float '[num_holes] =
--             UnsafeMkTensor . D.reshape [num_holes_] $ I.gather  -- F.squeeze 1
--                 (toDynamic hole_expansion_probs)
--                 rule_dim
--                 (toDynamic $ unsqueeze @1 rule_idx_by_hole)
--                 False
--     hole_idx :: Int = D.asValue $ F.argmax (F.Dim 0) F.RemoveDim $ toDynamic best_prob_by_hole
--     rule_idx :: Int = D.asValue $ D.select 0 hole_idx $ toDynamic rule_idx_by_hole

-- | fill a non-terminal leaf node in a PPT given hole/rule expansion probabilities
predictHole :: forall num_holes rules device . (StandardFloatingPointDTypeValidation device 'D.Float) => Bool -> [(String, Expr)] -> Expr -> Set String -> Tensor device 'D.Float '[num_holes, rules] -> IO (Expr, Set String)
predictHole randomHole variants ppt used hole_expansion_probs = do
    debug_ "predictHole"
    let (holes_dim, _rules_dim) :: (Int, Int) = (0, 1)
    [hole_idx, rule_idx] :: [Int] <- if randomHole then
            sampleIdxs . softmaxAll . toDynamic $ hole_expansion_probs
        else do
            let hole_idx :: Int = 0
            let holeScores :: Tensor device 'D.Float '[rules] = UnsafeMkTensor $ D.select holes_dim hole_idx $ toDynamic hole_expansion_probs
            let holeProbs  :: Tensor device 'D.Float '[rules] = softmax @0 holeScores
            [rule_idx] :: [Int] <- D.asValue <$> Distribution.sample (Categorical.fromProbs . toDynamic $ holeProbs) [1]
            return [hole_idx, rule_idx]
    -- order of rules: comes from `rule_emb`, which is just randomly assigned,
    -- so we can just arbitrarily associate this with any deterministic order e.g. that of `variants`
    debug_ $ "rule_idx: " <> show rule_idx
    let (_rule_str, rule_expr) :: (String, Expr) = variants !! rule_idx
    let block_name :: String = pp . head . fnAppNodes $ rule_expr
    debug_ $ "block_name: " <> show block_name
    let used' :: Set String = insert block_name used
    -- grab hole lenses by `findHolesExpr` and ensure `forwardPass` follows the same order to make these match.
    let (_hole_getter, hole_setter) :: (Expr -> Expr, Expr -> Expr -> Expr) =
            findHolesExpr ppt !! hole_idx
    let ppt' :: Expr = hole_setter ppt rule_expr
    debug_ . show $ (hole_idx, rule_idx, pp ppt')
    return (ppt', used')

-- | fill a random non-terminal leaf node as per `task_fn`
superviseHole :: forall device . (KnownDevice device, RandDTypeIsValid device 'D.Int64) => Bool -> HashMap String Expr -> Int -> Expr -> Expr -> IO Expr
superviseHole randomHole variantMap num_holes task_fn ppt = do
    debug_ "superviseHole"
    hole_idx :: Int <- if randomHole then do
            -- randomly pick a hole -- probably a good way to uniformly cover all scenarios...?
            -- technically holes closer to the root may go thru more rounds tho, creating more opportunities to randomly get picked
            hole_idx' :: Tensor device 'D.Int64 '[] <- randint 0 num_holes
            return $ toInt hole_idx'
        -- otherwise deterministically use the first hole
        else pure 0
    debug_ $ "hole_idx: " <> show hole_idx
    let (hole_getter, hole_setter) :: (Expr -> Expr, Expr -> Expr -> Expr) =
            findHolesExpr ppt !! hole_idx
    let rule_expr :: Expr = safeIndexHM variantMap . nodeRule . hole_getter $ task_fn
    debug_ $ "rule_expr: " <> pp rule_expr
    let ppt' :: Expr = hole_setter ppt rule_expr
    debug_ $ "ppt': " <> pp ppt'
    return ppt'

-- | supervise with task program to calculate the loss of the predicted hole/rule expansion probabilities for this PPT
fillHoleTrain :: forall num_holes rules device . (KnownDevice device, RandDTypeIsValid device 'D.Int64) => Bool -> HashMap String Expr -> HashMap String Int -> Expr -> Expr -> Tensor device 'D.Float '[num_holes, rules] -> IO (Expr, Tensor device 'D.Float '[num_holes])
fillHoleTrain randomHole variantMap ruleIdxs task_fn ppt hole_expansion_probs = do
    debug_ "fillHoleTrain"
    let (_hole_dim, rule_dim) :: (Int, Int) = (0, 1)
    let [num_holes, _rules] :: [Int] = shape' hole_expansion_probs
    debug_ $ "num_holes: " <> show num_holes
    ppt' :: Expr <- superviseHole @device randomHole variantMap num_holes task_fn ppt
    debug_ $ "ppt': " <> pp ppt'
    -- iterate over holes to get their intended expansion 'probabilities', used in calculating the loss
    let gold_rule_probs :: Tensor device 'D.Float '[num_holes] = UnsafeMkTensor . D.toDevice (deviceVal @device) . D.asTensor $ getGold . fst <$> findHolesExpr ppt
            where getGold = \gtr -> safeIndexHM ruleIdxs . nodeRule . gtr $ task_fn
    return (ppt', gold_rule_probs)

-- | calculate the loss by comparing the predicted expansions to the intended programs
calcLoss :: forall rules device shape synthesizer num_holes . (KnownDevice device, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, KnownNat rules, Synthesizer device shape rules synthesizer) => Bool -> HashMap String Expr -> Expr -> Tp -> HashMap String Int -> synthesizer -> Tensor device 'D.Float shape -> HashMap String Expr -> HashMap String Int -> HashMap String Int -> Int -> Bool -> [(String, Expr)] -> Interpreter (Tensor device 'D.Float '[])
calcLoss randomHole dsl task_fn taskType symbolIdxs model io_feats variantMap ruleIdxs variant_sizes max_holes maskBad variants = do
    debug "calcLoss"
    -- let (_hole_dim, rule_dim) :: (Int, Int) = (0, 1)
    -- (_program, golds, predictions, _filled) :: (Expr, [D.Tensor], [D.Tensor], Int) <- let
    --         fill = \(ppt, golds, predictions, filled) -> do
    --                 --  :: Tensor device 'D.Float '[num_holes, rules]
    --                 let predicted = predict @device @shape @rules @synthesizer model symbolIdxs ppt io_feats
    --                 -- debug $ "predicted: " <> show (shape' predicted)
    --                 predicted' <- if not maskBad then pure predicted else do
    --                     --  :: Tensor device 'D.Float '[num_holes, rules]
    --                     mask <-
    --                         fmap (Torch.Typed.Tensor.toDType @'D.Float . UnsafeMkTensor . D.asTensor) $
    --                         (\(hole_getter, hole_setter) -> mapM (fitExpr . hole_setter ppt . snd) variants)
    --                         `mapM` findHolesExpr ppt
    --                     return . Torch.Typed.Tensor.toDevice @device . asUntyped (F.mul $ toDynamic mask) $ predicted
    --                 -- debug $ "predicted': " <> show (shape' predicted')
    --                 (ppt', gold) <- liftIO $ fillHoleTrain randomHole variantMap ruleIdxs task_fn ppt predicted'
    --                 debug $ "ppt': " <> pp ppt'
    --                 return (ppt', (:) (toDynamic gold) $! golds, (:) (toDynamic predicted') $! predictions, filled + 1)
    --         in while_ (\(expr, _, _, filled) -> hasHoles expr && filled < max_holes) fill (letIn dsl (skeleton taskType), [], [], 0 :: Int)
    -- let gold_rule_probs :: D.Tensor = F.cat (F.Dim 0) golds
    -- -- debug $ "gold_rule_probs: " <> show (D.shape gold_rule_probs)
    -- let hole_expansion_probs :: D.Tensor = F.cat (F.Dim 0) predictions
    -- -- debug $ "hole_expansion_probs: " <> show (D.shape hole_expansion_probs)
    -- let loss :: Tensor device 'D.Float '[] = patchLoss @device @shape @rules model variant_sizes $ UnsafeMkTensor $ crossEntropy gold_rule_probs rule_dim hole_expansion_probs
    -- -- debug $ "loss: " <> show (shape' loss)
    let predicted = predict @device @shape @rules @synthesizer model symbolIdxs (letIn dsl (skeleton taskType)) io_feats
    let loss :: Tensor device 'D.Float '[] = patchLoss @device @shape @rules model variant_sizes $ (mulScalar (0.0 :: Float) $ sumAll $ io_feats) `add` (mulScalar (0.0 :: Float) $ sumAll $ predicted)
    return loss

-- | pre-calculate DSL stuff
prep_dsl :: TaskFnDataset -> PreppedDSL
prep_dsl TaskFnDataset{..} =
    PreppedDSL variants variant_sizes symbolIdxs ruleIdxs variantMap max_holes dsl'
    where
    variants :: [(String, Expr)] = (\(_k, v) -> (nodeRule v, v)) <$> exprBlocks
    variant_sizes :: HashMap String Int = fromList $ variantInt . snd <$> variants
    symbolIdxs :: HashMap String Int = indexList $ "undefined" : keys dsl
    ruleIdxs :: HashMap String Int = indexList $ fst <$> variants
    variantMap :: HashMap String Expr = fromList variants
    -- for synthesized programs, we apply the same maximum number of holes as used to generate this dataset. this allows our synthesizer enough power to construct the desired programs, while disallowing more complex programs than those of the maximum generated complexity. this choice is arbitrary; yet to think of a nicer solution.
    max_holes = maxHoles generationCfg
    -- DSL without entries equal to their key, for constructing let-in expressions.
    -- without this, blocks identical to their keys are seen as recursive, causing non-termination
    dsl' = filterWithKey (\k v -> k /= pp v) dsl

-- | train a NSPS model and return results
train :: forall device rules shape synthesizer . (KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat rules, KnownShape shape, Synthesizer device shape rules synthesizer, KnownNat (FromMaybe 0 (ExtractDim BatchDim shape))) => SynthesizerConfig -> TaskFnDataset -> synthesizer -> Interpreter [EvalResult]
train synthesizerConfig taskFnDataset init_model = do
    debug "train"
    let SynthesizerConfig{..} = synthesizerConfig
    let TaskFnDataset{..} = taskFnDataset

    -- GENERAL

    let [train_set, validation_set, test_set] :: [[(Expr, (Tp, Tp))]] = lists2pairs <$> (if cheat then replicate 3 . fstOf3 else untuple3) datasets
    let stdGen :: StdGen = mkStdGen seed
    liftIO $ setStdGen stdGen
    let init_lr :: Tensor device 'D.Float '[] = UnsafeMkTensor . D.asTensor $ learningRate
    let modelFolder = resultFolder <> "/" <> ppCfg synthesizerConfig
    liftIO $ createDirectoryIfMissing True modelFolder

    let prepped_dsl = prep_dsl taskFnDataset
    let PreppedDSL{..} = prepped_dsl
    debug $ "variants: " <> pp_ variants
    debug $ "variantMap: " <> pp_ variantMap
    debug $ "symbolIdxs: " <> show symbolIdxs
    debug $ "ruleIdxs: " <> show ruleIdxs

    -- MODELS
    let init_optim :: D.Adam = d_mkAdam 0 0.9 0.999 $ A.flattenParameters init_model
    let init_state = (stdGen, init_model, init_optim, False, [], init_lr, 0.0, 1)

    (_, model, _, _, eval_results, _, _, _) <- iterateLoopT init_state $ \ !state@(gen, model_, optim_, earlyStop, eval_results, lr, prev_acc, epoch) -> if earlyStop then exitWith state else do
        lift $ notice $ "epoch: " <> show epoch
        let (train_set', gen') = fisherYates gen train_set    -- shuffle
        let n :: Int = length train_set'
        pb <- lift . liftIO $ newProgressBar pgStyle 1 (Progress 0 n ("task-fns" :: Text))
        start <- lift . liftIO $ getCPUTime
        -- TRAIN LOOP
        (loss_train, model', optim', gen'', _) :: (Float, synthesizer, D.Adam, StdGen, Int) <- lift $ iterateLoopT (0.0, model_, optim_, gen', 0) $ \ !state@(train_loss, model, optim, gen_, task_fn_id) -> if task_fn_id >= n then exitWith state else do
                let task_fn_tp :: (Expr, (Tp, Tp)) = train_set' !! task_fn_id
                lift . info $ "task_fn_tp: \n" <> pp_ task_fn_tp
                let task_fn :: Expr = fst task_fn_tp
                -- lift . info $ "task_fn: \n" <> pp task_fn
                let tpInstPair :: (Tp, Tp) = snd task_fn_tp
                let taskType :: Tp = safeIndexHM fnTypes task_fn
                lift . info $ "taskType: " <> pp taskType
                let (target_tp_io_pairs, gen') :: (HashMap (Tp, Tp) [(Expr, Either String Expr)], StdGen) =
                        first (singleton tpInstPair) . fixSize (natValI @R3nnBatch) gen_ $ safeIndexHM (safeIndexHM fnTypeIOs task_fn) tpInstPair
                lift . info $ "target_tp_io_pairs: " <> pp_ target_tp_io_pairs
                --  :: Tensor device 'D.Float '[n'1, t * (2 * Dirs * h)]
                -- sampled_feats :: Tensor device 'D.Float '[R3nnBatch, t * (2 * Dirs * h)]
                let io_feats :: Tensor device 'D.Float shape = encode @device @shape @rules model target_tp_io_pairs
                -- lift . debug $ "io_feats: " <> show (shape' io_feats)
                loss :: Tensor device 'D.Float '[] <- lift $ calcLoss @rules randomHole dsl' task_fn taskType symbolIdxs model io_feats variantMap ruleIdxs variant_sizes max_holes maskBad variants
                -- lift . debug $ "loss: " <> show (shape' loss)
                -- TODO: do once for each mini-batch / fn?
                -- (newParam, optim') <- liftIO $ D.runStep model optim (toDynamic loss) $ toDynamic lr
                (newParam, optim') <- lift . liftIO $ doStep @device @shape @rules model optim loss lr
                let model' :: synthesizer = A.replaceParameters model newParam
                -- aggregating over task fns, which in turn had separately aggregated over any holes encountered across the different synthesis steps (so multiple times for a hole encountered across various PPTs along the way). this is fair, right?
                let train_loss' :: Float = train_loss + toFloat loss / fromIntegral n
                lift . liftIO $ incProgress pb 1
                return (train_loss', model', optim', gen', task_fn_id + 1)

        lift . debug $ "finished epoch training"
        end <- lift . liftIO $ getCPUTime
        let epochSeconds :: Double = (fromIntegral (end - start)) / (10^12)

        -- EVAL
        (earlyStop, eval_results', gen''') <- lift $ whenOrM (False, eval_results, gen'') (mod (epoch - 1) evalFreq == 0) $ do
            debug "evaluating"

            (acc_valid, loss_valid) <- evaluate @device @rules @shape taskFnDataset prepped_dsl bestOf maskBad randomHole model' validation_set

            say $ printf
                "Epoch: %03d. Train loss: %.4f. Validation loss: %.4f. Validation accuracy: %.4f.\n"
                epoch
                loss_train
                loss_valid
                acc_valid

            let modelPath = modelFolder <> printf "/%04d.pt" epoch
            liftIO $ D.save (D.toDependent <$> A.flattenParameters model') modelPath

            let eval_result = EvalResult epoch epochSeconds loss_train loss_valid acc_valid
            let eval_results' = (:) eval_result $! eval_results
            let earlyStop :: Bool = whenOr False (length eval_results' >= 2 * checkWindow) $ let
                    losses  :: [Float] = lossValid <$> eval_results'
                    losses' :: [Float] = take (2 * checkWindow) losses
                    (current_losses, prev_losses) = splitAt checkWindow losses'
                    current :: D.Tensor = F.mean . D.asTensor $ current_losses
                    prev    :: D.Tensor = F.mean . D.asTensor $ prev_losses
                    earlyStop :: Bool = D.asValue $ F.sub current prev `I.gtScalar` convergenceThreshold
                    in earlyStop
            when earlyStop $ debug "validation loss has converged, stopping early!"

            return $ (earlyStop, eval_results', gen'')

        let acc_valid :: Float = accValid $ head eval_results'
        -- decay the learning rate if accuracy decreases
        -- lr' :: Tensor device 'D.Float '[] <- case (acc_valid < prev_acc) of
        --     True -> do
        --         lift . info $ "accuracy decreased, decaying learning rate!"
        --         return . divScalar learningDecay $ lr
        --     False -> pure lr
        let lr' = lr

        return (gen''', model', optim', earlyStop, eval_results', lr', acc_valid, epoch + 1)

    -- write results to csv
    liftIO $ createDirectoryIfMissing True resultFolder
    let resultPath = resultFolder <> "/" <> ppCfg synthesizerConfig <> ".csv"
    let eval_results' = reverse eval_results -- we want the first epoch first
    liftIO $ BS.writeFile resultPath $ BS.packChars $ BL.unpackChars $ Csv.encodeByName evalResultHeader eval_results'
    info $ "data written to " <> resultPath

    return eval_results'

evaluate :: forall device rules shape synthesizer num_holes
          . ( KnownDevice device, RandDTypeIsValid device 'D.Float, MatMulDTypeIsValid device 'D.Float, SumDTypeIsValid device 'D.Float, BasicArithmeticDTypeIsValid device 'D.Float, RandDTypeIsValid device 'D.Int64, StandardFloatingPointDTypeValidation device 'D.Float, KnownNat rules, KnownShape shape, Synthesizer device shape rules synthesizer, KnownNat (FromMaybe 0 (ExtractDim BatchDim shape)))
         => TaskFnDataset -> PreppedDSL -> Int -> Bool -> Bool -> synthesizer -> [(Expr, (Tp, Tp))] -> Interpreter (Float, Float)
evaluate TaskFnDataset{..} PreppedDSL{..} bestOf maskBad randomHole model dataset = do
    debug "evaluate"
    let n :: Int = length dataset
    pb <- liftIO $ newProgressBar pgStyle 1 (Progress 0 n ("eval-fn" :: Text))
    (acc, loss, _) :: (Float, Float, Int) <- iterateLoopT (0.0, 0.0, 0) $ \ !state@(acc, loss, task_fn_id) -> if task_fn_id >= n then exitWith state else do
            let task_fn_tp :: (Expr, (Tp, Tp)) = dataset !! task_fn_id
            lift . debug $ "task_fn_tp: \n" <> pp_ task_fn_tp
            let task_fn :: Expr = fst task_fn_tp
            -- lift . debug $ "task_fn: \n" <> pp task_fn
            let tpInstPair :: (Tp, Tp) = snd task_fn_tp
            let taskType :: Tp = safeIndexHM fnTypes task_fn
            lift . debug $ "taskType: " <> pp taskType
            let target_tp_io_pairs :: HashMap (Tp, Tp) [(Expr, Either String Expr)] =
                    singleton tpInstPair $ safeIndexHM (safeIndexHM fnTypeIOs task_fn) tpInstPair
            let type_ins :: HashMap (Tp, Tp) [Expr] = fmap fst <$> target_tp_io_pairs
            lift . debug $ "type_ins: " <> pp_ type_ins
            let target_outputs :: [Either String Expr] =
                    fmap snd $ join . elems $ target_tp_io_pairs

            let io_feats :: Tensor device 'D.Float shape = encode @device @shape @rules model target_tp_io_pairs
            loss_ :: Tensor device 'D.Float '[] <- lift $ calcLoss @rules randomHole dsl' task_fn taskType symbolIdxs model io_feats variantMap ruleIdxs variant_sizes max_holes maskBad variants

            -- sample for best of 100 predictions
            -- TODO: dedupe samples before eval to save evals?
            -- TODO: consider A* / branch-and-bound / beam search instead
            -- pb <- liftIO $ newProgressBar pgStyle 1 (Progress 0 bestOf ("eval-samples" :: Text))
            sample_matches :: [Bool] <- lift $ replicateM bestOf $ do
                -- TODO: split io_feats and taskType based on param type instance combo 
                (program, used, _filled) :: (Expr, Set String, Int) <- let
                        --  :: (Int, Expr) -> IO (Int, Expr)
                        fill = \(ppt, used, filled) -> do
                                --  :: Tensor device 'D.Float '[num_holes, rules]
                                let predicted = predict @device @shape @rules @synthesizer model symbolIdxs ppt io_feats
                                debug $ "predicted: " <> show predicted
                                (ppt', used') <- liftIO $ predictHole randomHole variants ppt used predicted
                                return (ppt', used', filled + 1)
                        in while_ (\(ppt, used, filled) -> hasHoles ppt && filled < max_holes) fill (skeleton taskType, empty, 0 :: Int)
                debug $ pp program
                ok :: Bool <- if hasHoles program then pure False else do
                    let defs :: HashMap String Expr = pickKeysSafe (Data.Set.toList used) dsl'
                    let program' :: Expr = if null defs then program else letIn defs program

                    debug $ "type_ins: " <> pp_ type_ins
                    prediction_type_ios :: HashMap (Tp, Tp) [(Expr, Either String Expr)] <- let
                            compileInput :: (Tp, Tp) -> [Expr] -> Interpreter [(Expr, Either String Expr)] = \ tp_instantiation ins -> let
                                    n :: Int = length $ unTuple' $ ins !! 0
                                    -- crash_on_error=False is slower but lets me check if it compiles.
                                    in fnIoPairs False n program' tp_instantiation $ list ins
                            in sequence $ compileInput `mapWithKey` type_ins
                    debug $ "prediction_type_ios: " <> pp_ prediction_type_ios
                    let prediction_io_pairs :: [(Expr, Either String Expr)] =
                            join . elems $ prediction_type_ios
                    let outputs_match :: Bool = case length target_outputs == length prediction_io_pairs of
                            False -> False
                            True -> let
                                    prediction_outputs :: [Either String Expr] = snd <$> prediction_io_pairs
                                    output_matches :: [Bool] = uncurry (==) . mapBoth pp_ <$> target_outputs `zip` prediction_outputs
                                    in and output_matches
                    return outputs_match
                -- liftIO $ incProgress pb 1
                return ok

            let best_works :: Bool = or sample_matches
            -- let score :: Tensor device 'D.Float '[] = UnsafeMkTensor . F.mean . D.asTensor $ (fromBool :: (Bool -> Float)) <$> sample_matches
            let acc'  :: Float = acc  + fromBool best_works / fromIntegral n
            let loss' :: Float = loss + toFloat loss_      / fromIntegral n
            lift . liftIO $ incProgress pb 1
            return (acc', loss', task_fn_id + 1)

    return (acc, loss)
