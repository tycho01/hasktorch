{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE TupleSections #-}

-- | functions related to type generation
module Synthesis.TypeGen (module Synthesis.TypeGen) where

import System.Random (StdGen, mkStdGen, newStdGen)
import Control.Monad (join, replicateM)
import Data.HashMap.Lazy
  ( (!),
    HashMap,
    empty,
    insert,
    delete,
    mapWithKey,
    filterWithKey,
    fromListWith,
    keys,
    elems,
    toList,
    unionWith,
  )
import Data.Maybe (fromMaybe)
import Language.Haskell.Exts.Syntax
  ( Asst (..),
    Context (..),
    Promoted (..),
    TyVarBind (..),
    Type (..),
  )
import Util (thdOf3)
import Synthesis.Data hiding (nestLimit, maxInstances)
import Synthesis.Types
import Synthesis.Utility

-- | randomly generate a type
randomType :: HashMap Int [String] -> Bool -> Bool -> Int -> HashMap String [Tp] -> Int -> StdGen -> IO Tp
randomType tpsByArity allowAbstract allowFns nestLimit typeVars tyVarCount gen = do
    -- type variables
    -- TODO: allow generating new type vars
    let tyVarName :: String = "t" ++ show tyVarCount -- TODO: make this random?
    let tpVar :: IO Tp = return $ tyVar tyVarName
    let tpVars :: [IO Tp] = return . tyVar <$> keys typeVars
    let abstracts :: [IO Tp] = if allowAbstract then [tpVar] else []
    -- functions
    let gen_fn :: IO Tp = randomFnType tpsByArity allowAbstract allowFns nestLimit typeVars tyVarCount gen
    let fns :: [IO Tp] = [gen_fn | allowFns]
    -- base types
    let applied :: HashMap Int [IO Tp] = mapWithKey (\ i strs -> fillChildren gen i <$> strs) tpsByArity
    let base :: [IO Tp] = concat $ elems $ filterWithKey (\ k _v -> k == 0 || nestLimit > 0) applied
    -- total
    let options :: [IO Tp] = base ++ tpVars ++ abstracts ++ fns
    tp :: Tp <- snd $ pickG' gen options
    return tp
    where
      f = randomType tpsByArity allowAbstract allowFns (nestLimit - 1)
      fillChildren :: StdGen -> Int -> String -> IO Tp = \ g arity str -> do
        (g', tp) <- nest arity (\(gen_, a) -> do
          b <- f typeVars tyVarCount gen
          gen' <- newStdGen
          return $ (gen', tyApp a b)
          ) (g, tyCon str)
        return tp

-- | randomly generate a function type
-- | deprecated, not in actual use (`randomType` is only used with `allowFns` as `True` in another deprecated function)
-- TODO: ensure each type var is used at least twice
randomFnType :: HashMap Int [String] -> Bool -> Bool -> Int -> HashMap String [Tp] -> Int -> StdGen -> IO Tp
randomFnType tpsByArity allowAbstract allowFns nestLimit typeVars tyVarCount gen = do
  let f = randomType tpsByArity allowAbstract allowFns nestLimit
  gen' <- newStdGen
  tpIn :: Tp <- f typeVars tyVarCount gen
  let typeVarsIn :: HashMap String [Tp] = thdOf3 <$> findTypeVars tpIn
  let typeVars_ = mergeTyVars typeVars typeVarsIn
  tpOut :: Tp <- f typeVars_ tyVarCount gen'
  let fn :: Tp = tyFun tpIn tpOut
  return fn

-- merge two maps of type variables and their corresponding type constraints
-- | deprecated, not in actual use (`randomType` is only used with `allowFns` as `True` in another deprecated function)
mergeTyVars :: HashMap String [Tp] -> HashMap String [Tp] -> HashMap String [Tp]
mergeTyVars = unionWith $ \a b -> nubPp $ a ++ b

-- | helper function for `findTypeVars` and `findTypeVars_`
reconcileTypeVars :: (Int, Int, [Tp]) -> (Int, Int, [Tp]) -> (Int, Int, [Tp])
reconcileTypeVars (arity1, depth1, constraints1) (_arity2, depth2, constraints2) = (arity1, max depth1 depth2, constraints1 ++ constraints2)

-- | find the type variables and their constraints
findTypeVars :: Tp -> HashMap String (Int, Int, [Tp])
findTypeVars = fromListWith reconcileTypeVars . findTypeVars_ 0 0

-- | recursive `findTypeVars_` helper
findTypeVars_ :: Int -> Int -> Tp -> [(String, (Int, Int, [Tp]))]
findTypeVars_ arity depth tp =
  let f = findTypeVars_ arity $ depth + 1
   in case tp of
        TyForall _l maybeTyVarBinds maybeContext typ -> bindings ++ context ++ findTypeVars_ arity depth typ
          where
            bindings :: [(String, (Int, Int, [Tp]))] = toList $ fromListWith reconcileTypeVars $ (\(KindedVar _l name kind) -> (pp name, (arity, depth, [kind]))) <$> fromMaybe [] maybeTyVarBinds
            context :: [(String, (Int, Int, [Tp]))] = fromContext $ fromMaybe (CxEmpty l) maybeContext
            fromContext :: Context L -> [(String, (Int, Int, [Tp]))] = \case
              CxTuple _l assts -> concat $ unAsst <$> assts
              CxSingle _l asst -> unAsst asst
              CxEmpty _l -> []
            unAsst :: Asst L -> [(String, (Int, Int, [Tp]))] = \case
              TypeA _l tp' -> case tp' of
                TyApp _l a b -> [(pp b, (arity, depth, [a]))]
                _ -> f typ
              IParam _l _iPName a -> f a
              ParenA _l asst -> unAsst asst
              _ -> error "unsupported Assist"
        TyFun _l a b -> f a ++ f b
        TyTuple _l _boxed tps -> concat $ f <$> tps
        TyUnboxedSum _l tps -> concat $ f <$> tps
        TyList _l a -> f a
        TyParArray _l a -> f a
        TyApp _l a b -> findTypeVars_ (arity + 1) (depth + 1) a ++ f b
        TyVar _l _name -> [(pp tp, (arity, depth, []))]
        TyParen _l a -> findTypeVars_ arity depth a
        TyKind _l a kind -> f a ++ f kind
        TyPromoted _l promoted -> case promoted of
          PromotedList _l _bl tps -> concat $ f <$> tps
          PromotedTuple _l tps -> concat $ f <$> tps
          _ -> []
        TyEquals _l a b -> f a ++ f b
        TyBang _l _bangType _unpackedness a -> f a
        _ -> []

-- | substitute all type variable occurrences
fillTypeVars :: Tp -> HashMap String Tp -> Tp
fillTypeVars tp substitutions =
  let f = flip fillTypeVars substitutions
   in case tp of
        TyForall _l _maybeTyVarBinds _maybeContext a -> f a
        -- ^ if I'm filling type vars I guess type constraints can be stripped out
        TyFun _l a b -> tyFun (f a) $ f b
        TyTuple _l boxed tps -> TyTuple l boxed $ f <$> tps
        TyUnboxedSum _l tps -> TyUnboxedSum l $ f <$> tps
        TyList _l a -> tyList $ f a
        TyParArray _l a -> TyParArray l $ f a
        TyApp _l a b -> tyApp (f a) $ f b
        TyVar _l _name -> substitutions ! pp tp
        TyParen _l a -> TyParen l $ f a
        TyKind _l a kind -> TyKind l (f a) $ f kind
        TyPromoted _l promoted -> TyPromoted l $ case promoted of
          PromotedList _l bl tps -> PromotedList l bl $ f <$> tps
          PromotedTuple _l tps -> PromotedTuple l $ f <$> tps
          _ -> promoted
        TyEquals _l a b -> TyEquals l (f a) $ f b
        TyBang _l bangType unpackedness a -> TyBang l bangType unpackedness $ f a
        _ -> tp

-- | generate a number of monomorphic types to be used in type variable substitution
genTypes :: Int -> HashMap Int [String] -> Int -> Int -> IO (HashMap Int [Tp])
genTypes seed tpsByArity nestLimit maxInstances = do
  tps :: [Tp] <- nubPp . flatten <$> Many . fmap (One . pure) <$>
        -- replicateM maxInstances makeTp
        (makeTp . mkStdGen . (seed +)) `mapM` [1 .. maxInstances] 
  return . insert 0 tps . delete 0 $ fmap tyCon <$> tpsByArity
  where
    makeTp :: StdGen -> IO Tp =
        randomType tpsByArity False False nestLimit empty 0
