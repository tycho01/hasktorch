{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ImplicitParams #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TupleSections #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Synthesizer.Utility (module Synthesizer.Utility) where

import Prelude hiding (lookup, exp)
import System.Random (StdGen, mkStdGen)
import System.ProgressBar
import GHC.Stack
import GHC.TypeLits
import GHC.TypeNats (Nat, KnownNat, type (+), type (*))
import System.Random (RandomGen, Random, random)
import Debug.Trace (trace)
import Data.Int (Int64)
import Data.Maybe (fromJust)
import Data.List (findIndex, cycle)
import Data.Foldable (toList, foldl', foldr')
import Data.Monoid
import Data.Hashable (Hashable)
import Data.HashMap.Lazy (HashMap, fromList, lookup, filterWithKey)
import Data.Proxy
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Aeson as Aeson
import qualified Data.Csv as Csv
import qualified Data.ByteString               as BS
import qualified Data.ByteString.Internal      as BS
import qualified Data.ByteString.Lazy.Internal as BL
import System.Environment (getEnv)
import Control.Exception (SomeException, try, assert)
import Control.Applicative
import Control.Monad (void, foldM, (=<<))
import Language.Haskell.Interpreter (Interpreter)
import Language.Haskell.Exts.Syntax

import           Torch.Typed.Aux
import           Torch.Typed.Tensor hiding (dim)
import qualified Torch.Typed.Tensor
import           Torch.Typed.Functional hiding (trace)
import           Torch.Typed.Parameter
import qualified Torch.Typed.Parameter
import           Torch.Typed.NN
import           Torch.Typed.NN.Recurrent.LSTM
import           Torch.HList
import           Torch.Scalar
import qualified Torch.NN                      as A
import           Torch.Autograd                as D
import           Torch.TensorFactories         as D
import qualified Torch.Tensor                  as D
import qualified Torch.DType                   as D
import qualified Torch.Device                  as D
import qualified Torch.Optim                   as D
import qualified Torch.Functional.Internal     as I
import qualified Torch.Functional              as F
import qualified Torch.Internal.Class                    as ATen
import qualified Torch.Internal.Managed.Native           as ATen
import qualified Torch.Internal.Unmanaged.Type.Context   as ATen
import qualified Torch.Distributions.Distribution as Distribution
import qualified Torch.Distributions.Categorical as Categorical

import Synthesis.Data
import Synthesis.Orphanage ()
import Synthesis.Utility
import Synthesis.Ast (genBlockVariants)
import Synthesis.Hint

import System.IO.Unsafe (unsafePerformIO)
import Torch.Internal.Cast

import System.IO.Unsafe
import qualified Torch.Internal.Managed.Native as ATen
import qualified Torch.Internal.Managed.Type.Tensor as ATen
import qualified Torch.Internal.Managed.Cast
import Torch.Internal.Cast

d_repeat :: [Int] -> D.Tensor -> D.Tensor
d_repeat ns t = unsafePerformIO $ (cast2 ATen.tensor_repeat_l) t ns

repeatDim :: Int -> Int -> D.Tensor -> D.Tensor
repeatDim dim n t = d_repeat (replicate dim 1 <> [n] <> replicate (D.dim t - dim) 1) $ I.unsqueeze t dim

type Dir = 'Bidirectional
type Dirs = NumberOfDirections Dir
dirs :: Int
dirs = natValI @Dirs

type Cpu = '( 'D.CPU, 0)
type Gpu = '( 'D.CUDA, 0)

cpu :: D.Device = D.Device D.CPU  0
gpu :: D.Device = D.Device D.CUDA 0

getDevice :: IO D.Device
getDevice = do
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  print $ show deviceStr
  -- DEVICE: getEnv: does not exist (no environment variable)
  return $ case deviceStr of
    Right "cuda:0" -> D.Device D.CUDA 0
    Right "cpu"    -> D.Device D.CPU 0
    _              -> D.Device D.CPU 0

-- cpu = Proxy @'( 'D.CPU, 0)

-- cuda0 = Proxy @'( 'D.CUDA, 0)

-- | any available devices
availableDevices :: [D.Device]
availableDevices =
  [D.Device { D.deviceType = D.CPU, D.deviceIndex = 0 }]
    <> (if hasCuda
          then [D.Device { D.deviceType = D.CUDA, D.deviceIndex = 0 }]
          else mempty
        )

-- | check if we can use GPU
hasCuda :: Bool
hasCuda = unsafePerformIO $ cast0 ATen.hasCUDA

-- | get the device to run on: CUDA if available, otherwise CPU
fastestDevice :: (D.DeviceType, Nat)
fastestDevice = (if hasCuda then D.CUDA else D.CPU, 0)

-- | right-pad a list to a given length
padRight :: a -> Int -> [a] -> [a]
padRight c n xs = xs ++ replicate (n - length xs) c

-- | use an untyped fn on a typed tensor, works tho unsafe
asUntyped :: forall device dtype shape device' dtype' shape'
           . (D.Tensor -> D.Tensor)
          -> Tensor device dtype shape
          -> Tensor device' dtype' shape'
asUntyped f = UnsafeMkTensor . f . toDynamic

-- | run an untyped op on a typed tensor, delaying safety checks to run-time
asUntyped' :: forall device dtype shape device' dtype' shape'
          .  (TensorOptions shape' dtype' device')
          => (D.Tensor -> D.Tensor)
          -> Tensor device dtype shape
          -> Tensor device' dtype' shape'
asUntyped' f tensor = let
        untyped = f . toDynamic $ tensor
        tensor' = UnsafeMkTensor untyped
        check = D.shape . toDynamic
        gold = optionsRuntimeShape @shape' @dtype' @device'
        in assertEqBy check gold tensor'

-- | lift a tensor operation for use on tensors on any device. useful for models that don't understand devices.
asCPU :: (D.Tensor -> D.Tensor) -> D.Tensor -> D.Tensor
asCPU f t = D.toDevice (D.device t) . f . D.toDevice (D.Device D.CPU 0) $ t

-- | get the run-time shape of a typed tensor
shape' :: Tensor device dtype shape -> [Int]
shape' = D.shape . toDynamic

-- | stack alternative with a nicer argument order
stack' :: Int -> [D.Tensor] -> D.Tensor
stack' = flip I.stack

-- | cast Int to Int64 (i.e. Torch's Long)
asLong :: Int -> Int64
asLong = fromIntegral

-- | `select` alternative that retains the dimension as a 1
-- | I want this as a built-in, see https://github.com/pytorch/pytorch/issues/34788
select' :: D.Tensor -> Int -> Int -> D.Tensor
select' tensor dim idx = D.indexSelect dim idxs tensor
        where idxs = D.toDevice (D.device tensor) . D.asTensor $ [asLong idx]

-- | point-free untyped select
select'' :: Int -> Int -> D.Tensor -> D.Tensor
select'' dim idx tensor = select' tensor dim idx

-- | spread the given dimension from a D.Tensor out as a list (keepdim=True)
unDim :: Int -> D.Tensor -> [D.Tensor]
unDim dim tensor = select' tensor dim <$> [0 .. (D.shape tensor !! dim) - 1]

-- | intended as a PoC for NSPS's cross-correlation encoder logic. unused.
rotate :: [Float] -> [[Float]]
rotate r = res
    where
        n :: Int = length r
        n' :: Int = 2 * n - 1
        l :: [Float] = padRight (0.0 :: Float) n' r
        is :: [Int] = [0 .. length l - 1]
        res :: [[Float]] = (\i -> (\j -> l !! (mod (j - i) n') ) <$> is) <$> is

-- rotateT :: forall n device dtype . KnownNat n => Tensor device dtype '[n] -> [[D.Tensor]]
-- rotateT r = let
--         -- n :: Int = shape r !! 0
--         n = natValI @n
--         -- type N' = 2 * n - 1
--         n' :: Int = 2 * n - 1
--         -- type N = 2 -- ??
--         l :: Tensor device dtype '[n + (n - 1)] = constantPadNd1d @'(0, n-1) 0.0 r
--         is :: [Int] = [0 .. n'-1]
--         -- stack :: [Tensor] -> Int -> Tensor
--         res :: [[D.Tensor]] = (\i -> (\j -> D.select 0 (mod (j - i) n') $ toDynamic l) <$> is) <$> is
--         -- stack @0 (t :. HNil)
--         -- res = (\i -> (\j -> select @0 l @(GHC.TypeNats.Mod (j - i) (2 * n - 1)) l) <$> is) <$> is
--     in res

-- | extract an App chain from an App
fnAppNodes :: Expr -> [Expr]
fnAppNodes = \app -> case app of
    Paren _l xpr -> f xpr
    Let _l _binds xpr -> f xpr
    ExpTypeSig _l xpr _tp -> f xpr
    App _l a b -> f a ++ [b]
    Con _l _qname -> [app]
    Var _l _qname -> [app]
    _ -> []
    where f = fnAppNodes

-- | Let R(n) represent the production rule of non-leaf node n∈N.
nodeRule :: Expr -> String
nodeRule = appRule . fnAppNodes

-- | serialize a function application chain as a skeleton
appRule :: [Expr] -> String
appRule = \case
            [] -> error "no fn!"
            fn : args -> unwords $ pp fn : replicate (length args) "_"

lookupRule :: (Eq k, Hashable k, Show k) => HashMap k v -> k -> v
lookupRule hm k = case (lookup k hm) of
    Just x -> x
    Nothing -> error $ "the DSL does not contain rule " ++ show k ++ "!"

-- | get holed variants for a DSL
-- | deprecated, not in use
dslVariants :: HashMap String Expr -> Interpreter [(String, Expr)]
dslVariants dsl = do
    fn_types :: HashMap String Tp <- exprType `mapM` dsl
    return $ genBlockVariants fn_types

-- | split a (batch-first) tensor into batches (zero-padded for the last one)
batchTensor :: Int -> D.Tensor -> [D.Tensor]
batchTensor batch_size tensor = let
    nDim :: Int = 0
    n :: Int = D.size nDim tensor
    numBatches :: Int = ((n-1) `div` batch_size) + 1
    diff :: Int = numBatches * batch_size - n
    paddings :: [Int] = replicate (2 * D.dim tensor - 1) 0 <> [diff]
    tensor' :: D.Tensor = F.constantPadNd1d paddings 0.0 tensor
    f :: Int -> D.Tensor = \i -> let
            from :: Int = i * batch_size
            to   :: Int = (i+1) * batch_size - 1
            idxs :: [Int] = [from .. to]
            idxs' = D.toDevice (D.device tensor) . D.asTensor $ asLong <$> idxs
        in D.indexSelect nDim idxs' tensor'
    in f <$> [0 .. numBatches-1]

-- -- | statically typed version of batchTensor
-- -- | deprecated, not in use
-- batchTensor'
--     :: forall batchSize dim device dtype shape shape'
--      . ( KnownNat batchSize
--        , KnownShape shape
--        , TensorOptions shape dtype device
--        , dim ~ 0
--        , shape' ~ FromJust (ReplaceDim dim shape batchSize)
--        )
--     => Tensor device dtype shape
--     -> [Tensor device dtype shape']
-- batchTensor' tensor = let
--     batch_size = natValI @batchSize
--     nDim = natValI @dim
--     n :: Int = D.size nDim (toDynamic tensor)
--     numBatches :: Int = ((n-1) `div` batch_size) + 1
--     diff :: Int = numBatches * batch_size - n
--     paddings :: [Int] = replicate (2 * Torch.Typed.Tensor.dim tensor - 1) 0 <> [diff]
--     tensor' :: D.Tensor = F.constantPadNd1d paddings 0.0 $ toDynamic tensor
--     f :: Int -> Tensor device dtype shape' = \i -> let
--             from :: Int = i * batch_size
--             to   :: Int = (i+1) * batch_size - 1
--             idxs :: [Int] = [from .. to]
--         in UnsafeMkTensor $ D.indexSelect nDim (D.toDevice (D.device $ toDynamic tensor) . D.asTensor $ asLong <$> idxs) tensor'
--     in f <$> [0 .. numBatches-1]

-- type family FromJust (maybe :: Maybe a) :: a where
--   FromJust (Just a) = a
-- --   FromJust Nothing = Nothing

type family FromMaybe (def :: a) (maybe :: Maybe a) :: a where
  FromMaybe _def (Just a) = a
  FromMaybe def Nothing = def

-- -- | deprecated, not in use
-- toMaybe :: Bool -> a -> Maybe a
-- toMaybe False _ = Nothing
-- toMaybe True  x = Just x

whenOr :: a -> Bool -> a -> a
-- whenOr def cond x = fromMaybe def $ toMaybe cond x
whenOr def cond x = if cond then x else def

whenOrM :: (Applicative m) => a -> Bool -> m a -> m a
whenOrM def cond x = if cond then x else pure def

-- | shuffle a tensor in a given dimension
-- | deprecated, not in use
shuffle :: forall g . (RandomGen g) => g -> Int -> D.Tensor -> (g, D.Tensor)
shuffle gen dim tensor = (gen', shuffled)
    where
        n = D.size dim tensor
        idxs = [0 .. n-1]
        (idxs', gen') = fisherYates gen idxs
        idxs'' = D.toDevice (D.device tensor) . D.asTensor $ asLong <$> idxs'
        shuffled = D.indexSelect dim idxs'' tensor

-- | square a tensor, for use in mean-square-error loss
square :: Tensor device 'D.Float shape -> Tensor device 'D.Float shape
square = pow (2 :: Int)

-- | cumulative fold
scan :: (Foldable t) => a -> (a -> a -> a) -> t a -> [a]
scan acc f = tail . foldl' (\ as a -> as <> [f a (last as)]) [acc]

-- | get cumulative probabilities
cumulative :: (Num a, Foldable t) => t a -> [a]
cumulative = scan 0 (+)

-- | randomly pick an item by relative probabilities (should sum to 1).
categorical :: (RandomGen g, Fractional a, Ord a, Random a, Foldable t) => g -> t a -> Int
categorical gen probs =
    fromJust . findIndex (> x) $ cumulative probs
    where (x, _gen') = random gen

-- | make an assertion thru a predicate
assertP :: (Show a) => (a -> Bool) -> a -> a
assertP pred_fn x = case pred_fn x of
    True -> x
    False -> error $ "assertP failed on input: " <> show x

-- | assert an equality check by a mapper function
assertEqBy :: (Show b, Eq b) => (a -> b) -> b -> a -> a
assertEqBy fn gold x = let x' = fn x in case x' == gold of
    True -> x
    False -> error $ "equality check failed on input ( " <> show x' <> " ) with gold value ( " <> show gold <> " )"

-- | assert an equality check -- yields a nicer stack trace than assertP
assertEq :: (Show a, Eq a) => a -> a -> a
assertEq = assertEqBy id

-- | apply a softmax over all dimensions
softmaxAll :: D.Tensor -> D.Tensor
softmaxAll t = D.reshape (D.shape t) $ F.softmax (F.Dim 0) $ F.flattenAll t

-- | like np.unravel_idx, unravel a flat index (from e.g. argmax_t) to the dimensions of a tensor
unravelIdx :: D.Tensor -> Int -> [Int]
unravelIdx t idx = snd . foldr' (\ dim_ (idx_, idxs) -> (idx_ `Prelude.div` dim_, idx_ `Prelude.mod` dim_ : idxs)) (idx, []) $ D.shape t

-- TODO: replace with built-in
-- TODO: strip device moving off once nllLoss' patch gets in
-- | calculate the cross-entropy loss given target indices, a class dimension, and a predictions tensor
crossEntropy :: D.Tensor -> Int -> D.Tensor -> D.Tensor
crossEntropy target dim input = D.toDevice (D.device target) $ F.nllLoss' (D.toDevice cpu target) $ F.logSoftmax (F.Dim dim) $ D.toDevice cpu $ input

-- | adjusted Torch.Typed.NN.Recurrent.LSTM.lstm to dynamically calculate batch size
-- | TODO: just batch inputs, ensuring dummy items won't influence results?
lstmDynamicBatch
  :: forall
       shapeOrder
       batchSize
       seqLen
       directionality
       initialization
       numLayers
       inputSize
       outputSize
       hiddenSize
       inputShape
       outputShape
       hxShape
       parameters
       tensorParameters
       dtype
       device
   . ( KnownNat (NumberOfDirections directionality)
     , KnownNat numLayers
     -- , KnownNat batchSize
     , KnownNat hiddenSize
     , KnownRNNShapeOrder shapeOrder
     , KnownRNNDirectionality directionality
     , outputSize ~ (hiddenSize * NumberOfDirections directionality)
     , inputShape ~ RNNShape shapeOrder seqLen batchSize inputSize
     , outputShape ~ RNNShape shapeOrder seqLen batchSize outputSize
     , hxShape ~ '[numLayers * NumberOfDirections directionality, batchSize, hiddenSize]
     , Parameterized (LSTM inputSize hiddenSize numLayers directionality dtype device) parameters
     , tensorParameters ~ LSTMR inputSize hiddenSize numLayers directionality dtype device
     , ATen.Castable (HList tensorParameters) [D.ATenTensor]
     , HMap' ToDependent parameters tensorParameters
     )
  => Bool
  -> LSTMWithInit
       inputSize
       hiddenSize
       numLayers
       directionality
       initialization
       dtype
       device
  -> Tensor device dtype inputShape
  -> ( Tensor device dtype outputShape
     , Tensor device dtype hxShape
     , Tensor device dtype hxShape
     )
lstmDynamicBatch dropoutOn (LSTMWithConstInit lstm@(LSTM _ (Dropout dropoutProb)) cc hc) input
  = Torch.Typed.Functional.lstm
    @shapeOrder
    @directionality
    @numLayers
    @seqLen
    @batchSize
    @inputSize
    @outputSize
    @hiddenSize
    @inputShape
    @outputShape
    @hxShape
    @tensorParameters
    @dtype
    @device
    (hmap' ToDependent . flattenParameters $ lstm)
    dropoutProb
    dropoutOn
    (cc', hc')
    input
 where
  cc' =
    asUntyped (
      D.reshape [natValI @(numLayers * NumberOfDirections directionality), batchSize, natValI @hiddenSize]
      . (\t -> F.expand t False [batchSize, natValI @(numLayers * NumberOfDirections directionality), natValI @hiddenSize])
      )
      $ cc
  hc' =
    asUntyped (
      D.reshape [natValI @(numLayers * NumberOfDirections directionality), batchSize, natValI @hiddenSize]
      . (\t -> F.expand t False [batchSize, natValI @(numLayers * NumberOfDirections directionality), natValI @hiddenSize])
      )
      $ hc
  batchSize = D.size (if rnnBatchFirst @shapeOrder then 0 else 1) $ toDynamic input

d_mkAdam
  :: Int
  -> Float
  -> Float
  -> [A.Parameter]
  -> D.Adam
d_mkAdam iter beta1 beta2 parameters =
    D.Adam
        beta1
        beta2
        (D.zerosLike . D.toDependent <$> parameters)
        (D.zerosLike . D.toDependent <$> parameters)
        iter

instance (Foldable t, Traversable t, A.Parameterized a) => A.Parameterized (t a) where
  flattenParameters = (=<<) A.flattenParameters . toList
  replaceOwnParameters = mapM A.replaceOwnParameters

instance A.Parameterized (Parameter device dtype shape) where
  flattenParameters (UnsafeMkParameter param) = pure param
  replaceOwnParameters _ = UnsafeMkParameter <$> A.nextParameter

-- | sample (with replacement) from a tensor in a given dimension
sampleTensorWithReplacement :: forall dim size device dtype shape' . (KnownNat dim, KnownNat size, TensorOptions shape' dtype device, KnownShape shape') => Int -> D.Tensor -> IO (Tensor device dtype shape')
sampleTensorWithReplacement n tensor = do
    sampled_idxs :: D.Tensor <- D.toDevice (D.device tensor) . F.toDType D.Int64 <$> D.randintIO' 0 n [natValI @size]
    return . UnsafeMkTensor $ D.indexSelect (natValI @dim) sampled_idxs tensor

-- | sample (without replacement, with pool resetting) from a tensor in a given dimension
sampleTensorWithoutReplacement :: forall dim size device dtype shape' . (KnownNat dim, KnownNat size, TensorOptions shape' dtype device, KnownShape shape') => StdGen -> Int -> D.Tensor -> (StdGen, Tensor device dtype shape')
sampleTensorWithoutReplacement gen n tensor = (gen', t) where
    (idxs', gen') = fisherYates gen $ [0 .. n - 1]
    idxs :: [Int] = take (natValI @size) . cycle $ idxs'
    sampled_idxs :: D.Tensor = D.toDevice (D.device tensor) . F.toDType D.Int64 . D.asTensor $ idxs
    t = UnsafeMkTensor $ D.indexSelect (natValI @dim) sampled_idxs tensor

-- | pretty-print a configuration for use in file names of result files, which requires staying within a 256-character limit.
ppCfg :: Aeson.ToJSON a => a -> String
ppCfg cfg = replacements [("\"",""),("\\",""),("/","\\"),("false","0"),("true","1"),("learningRate","lr"),("convergenceThreshold","threshold"),("learningDecay","lrDecay")] . show . Aeson.encode . filterWithKey (\ k _v -> k `Set.notMember` Set.fromList (Text.pack <$> ["verbosity","resultFolder","regularization","savedModelPath","initialEpoch"])) . fromJust $ (Aeson.decode (Aeson.encode cfg) :: Maybe Aeson.Object)

-- https://hackage.haskell.org/package/relude-0.6.0.0/docs/Relude-Extra-Tuple.html#v:traverseToSnd
traverseToSnd :: Functor t => (a -> t b) -> a -> t (a, b)
traverseToSnd f a = (a,) <$> f a

liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = liftA3 f a b c <*> d

liftA5 :: Applicative f => (a -> b -> c -> d -> e -> f') -> f a -> f b -> f c -> f d -> f e -> f f'
liftA5 f a b c d e = liftA4 f a b c d <*> e

liftA6 :: Applicative f => (a -> b -> c -> d -> e -> f' -> g) -> f a -> f b -> f c -> f d -> f e -> f f' -> f g
liftA6 f a b c d e f' = liftA5 f a b c d e <*> f'

liftA7 :: Applicative f => (a -> b -> c -> d -> e -> f' -> g -> h) -> f a -> f b -> f c -> f d -> f e -> f f' -> f g -> f h
liftA7 f a b c d e f' g = liftA6 f a b c d e f' <*> g

-- | calculate a cartesian product, used for hyper-parameter combinations
cartesianProduct6 :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [(a, b, c, d, e, f)]
cartesianProduct6 = liftA6 (,,,,,)

uncurry6 :: (a -> b -> c -> d -> e -> f' -> g) -> (a, b, c, d, e, f') -> g
uncurry6 f ~(a, b, c, d, e, f') = f a b c d e f'

writeCsv :: Csv.ToNamedRecord a => FilePath -> Csv.Header -> [a] -> IO ()
writeCsv filePath header =
    BS.writeFile filePath . BS.packChars . BL.unpackChars . Csv.encodeByName header

-- | take any given indices of a list
pickIdxs :: [Int] -> [a] -> [a]
pickIdxs idxs lst = (lst !!) <$> idxs

-- | print info about an item
-- | not referentially transparent, see https://hackage.haskell.org/package/base-4.14.0.0/docs/Debug-Trace.html#v:trace
showTrace :: Show a => a -> a
showTrace a = trace (show a) a

-- | print info about an item with a description
showTraceOf :: Show a => String -> a -> a
showTraceOf str a = trace (str <> ": " <> show a) a

-- | use a Categorical distribution to sample indices from a probability tensor
sampleIdxs :: D.Tensor -> IO [Int]
sampleIdxs t = do
    let ps :: D.Tensor = flip I.unsqueeze 0 . F.flattenAll $ t
    [[idx]] :: [[Int]] <- D.asValue <$> Distribution.sample (Categorical.fromProbs ps) [1]
    return $ unravelIdx t idx

pickDataset :: Tpl3 (HashMap Expr [(Tp, Tp)]) -> String -> [(Expr, (Tp, Tp))]
pickDataset datasets dataset_str = dataset where
    (train_set, validation_set, test_set) :: Tpl3 [(Expr, (Tp, Tp))] = mapTuple3 lists2pairs datasets
    dataset = case dataset_str of
        "training" -> train_set
        "validation" -> validation_set
        "test" -> test_set
        x -> error $ "dataset " <> x <> " not recognized!"

clipGradients' :: Float -> D.Gradients -> D.Gradients
clipGradients' v (D.Gradients gradients) = D.Gradients $ F.clamp (-v) v <$> gradients

decayWeights' :: (Scalar a, Num a) => a -> [D.IndependentTensor] -> D.Gradients -> D.Gradients
decayWeights' v parameters (D.Gradients gradients) = D.Gradients $ zipWith (\ param gradient -> (if v == 0.0 then id else F.add (F.mulScalar v param)) gradient) parameters gradients
