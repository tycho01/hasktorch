{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}

module Torch.Tensor where

import Control.Monad (forM_, forM)
import Control.Exception.Safe (throwIO)
import Foreign.ForeignPtr
import Foreign.Ptr
import Foreign.Storable
import Foreign.C.Types
import System.IO.Unsafe
import Data.Int (Int64)
import Data.List (intercalate)
import Data.Proxy
import Data.Reflection
import Numeric

import ATen.Cast
import ATen.Class (Castable(..), CppTuple2(..), CppTuple3(..), CppTuple4(..))
import qualified ATen.Unmanaged.Type.Tensor as Unmanaged (tensor_data_ptr)
import qualified ATen.Managed.Type.Tensor as ATen
import qualified ATen.Managed.Type.TensorOptions as ATen
import qualified ATen.Managed.Type.StdArray as ATen
import qualified ATen.Managed.Type.StdString as ATen
import qualified ATen.Managed.Native as ATen
import qualified ATen.Managed.Cast as ATen
import qualified ATen.Type as ATen
import qualified ATen.Const as ATen
import qualified Torch.Managed.Native as LibTorch

import Torch.DType
import Torch.TensorOptions


type ATenTensor = ForeignPtr ATen.Tensor

-- do not use the constructor
newtype Tensor = Unsafe ATenTensor

instance Castable Tensor ATenTensor where
  cast (Unsafe aten_tensor) f = f aten_tensor
  uncast aten_tensor f = f $ Unsafe aten_tensor

--------------------------------------------------------------------------------
-- Basic tensor properties
--------------------------------------------------------------------------------

numel :: Tensor -> Int
numel t = unsafePerformIO $ cast1 ATen.tensor_numel $ t

size :: Tensor -> Int -> Int
size t dim = unsafePerformIO $ (cast2 ATen.tensor_size_l) t dim

shape :: Tensor -> [Int]
shape t = unsafePerformIO $ (cast1 ATen.tensor_sizes) t

dim :: Tensor -> Int
dim t = unsafePerformIO $ (cast1 ATen.tensor_dim) t

dtype :: Tensor -> DType
dtype t = unsafePerformIO $ (cast1 ATen.tensor_scalar_type) t

toDouble :: Tensor -> Double
toDouble t = unsafePerformIO $ cast1 ATen.tensor_item_double $ t

toInt :: Tensor -> Int
toInt t = unsafePerformIO $ cast1 ATen.tensor_item_int64_t $ t

select :: Tensor -> Int -> Int -> Tensor
select t dim idx = unsafePerformIO $ (cast3 ATen.tensor_select_ll) t dim idx

reshape :: Tensor -> [Int] -> Tensor
reshape t shape = unsafePerformIO $ (cast2 ATen.reshape_tl) t shape

--------------------------------------------------------------------------------
-- Move backend
--------------------------------------------------------------------------------

toSparse :: Tensor -> Tensor
toSparse t = unsafePerformIO $ (cast1 ATen.tensor_to_sparse) t

toDense :: Tensor -> Tensor
toDense t = unsafePerformIO $ (cast1 ATen.tensor_to_dense) t

toCPU :: Tensor -> Tensor
toCPU t = unsafePerformIO $ (cast1 ATen.tensor_cpu) t

toCUDA :: Tensor -> Tensor
toCUDA t = unsafePerformIO $ (cast1 ATen.tensor_cuda) t

--------------------------------------------------------------------------------
-- Indexing support
--------------------------------------------------------------------------------

(@@) :: TensorIndex a => Tensor -> a -> Tensor
t @@ idx = fst $ indexWith (t, 0) idx

class TensorIndex a where
  indexWith :: (Tensor, Int) -> a -> (Tensor, Int)

instance TensorIndex Int where
  indexWith (t, offset) idx = (select t offset idx, offset)

instance TensorIndex Integer where
  indexWith (t, offset) idx = (select t offset (fromIntegral idx), offset)

instance TensorIndex () where
  indexWith (t, offset) idx = (t, offset + 1)

instance (TensorIndex a, TensorIndex b) => TensorIndex (a,b) where
  indexWith toff (a, b) = indexWith (indexWith toff a) b

instance (TensorIndex a, TensorIndex b, TensorIndex c) => TensorIndex (a,b,c) where
  indexWith toff (a, b, c) = indexWith (indexWith (indexWith toff a) b) c

instance (TensorIndex a, TensorIndex b, TensorIndex c, TensorIndex d) => TensorIndex (a,b,c,d) where
  indexWith toff (a, b, c, d) = indexWith (indexWith (indexWith (indexWith toff a) b) c) d

--------------------------------------------------------------------------------
-- Scalar <-> Tensor promotion
--------------------------------------------------------------------------------

class TensorLike a where
  asTensor' :: a -> TensorOptions -> Tensor
  asTensor :: a -> Tensor
  asValue :: Tensor -> a
  -- Internal functions(like "_xxx") are below. Do not use them directly.
  _dtype :: DType
  _dims :: a -> [Int]
  _peekElemOff :: Ptr () -> Int -> [Int] -> IO a
  _pokeElemOff :: Ptr () -> Int -> a -> IO ()

int64_opts = withDType Int64 defaultOpts
float_opts = withDType Float defaultOpts

withTensor :: Tensor -> (Ptr () -> IO a) -> IO a
withTensor t fn = cast t $ \t' -> withForeignPtr t' $ \tensor_ptr -> Unmanaged.tensor_data_ptr tensor_ptr >>= fn

instance (Reifies a DType, Storable a) => TensorLike a where
  asTensor' v opts = unsafePerformIO $ do
    t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) [] $ withDType (_dtype @a) opts
    withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  asTensor v = asTensor' v defaultOpts

  asValue t = unsafePerformIO $ do
    if _dtype @a == dtype t
    then do
      withTensor t $ \ptr -> do
        _peekElemOff ptr 0 []
    else
      throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a)  ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = reflect (Proxy :: Proxy a)
  _dims _ = []
  _peekElemOff ptr offset _ = peekElemOff (castPtr ptr) offset
  _pokeElemOff ptr offset v = pokeElemOff (castPtr ptr) offset v

instance {-# OVERLAPPING #-}TensorLike a => TensorLike [a] where
  asTensor' v opts = unsafePerformIO $ do
    t <- ((cast2 LibTorch.empty_lo) :: [Int] -> TensorOptions -> IO Tensor) (_dims v) $ withDType (_dtype @a) opts
    withTensor t $ \ptr -> do
      _pokeElemOff ptr 0 v
    return t

  asTensor v = asTensor' v defaultOpts

  asValue t = unsafePerformIO $ do
    if _dtype @a == dtype t
    then do
      withTensor t $ \ptr -> do
        _peekElemOff ptr 0 (shape t)
    else
      throwIO $ userError $ "The infered DType of asValue is " ++ show (_dtype @a)  ++ ", but the DType of tensor on memory is " ++ show (dtype t) ++ "."

  _dtype = _dtype @a

  _dims [] = []
  _dims v@(x:_) = (length v):(_dims x)

  _peekElemOff ptr offset [] = return []
  _peekElemOff ptr offset (d:dims) =
    let width = product dims
    in forM [0..(d-1)] $ \i ->
         _peekElemOff ptr (offset+i*width) dims

  _pokeElemOff ptr offset [] = return ()
  _pokeElemOff ptr offset v@(x:_) =
    let width = product (_dims x)
    in forM_ (zip [0..] v) $ \(i,d) ->
         if product (_dims d) == width -- This validation may be slow.
         then (_pokeElemOff @a) ptr (offset+i*width) d
         else throwIO $ userError $ "There are lists having different length."

--------------------------------------------------------------------------------
-- Show
--------------------------------------------------------------------------------

instance Show Tensor where
  show t = case (dim t) of
      0 -> details ++ show0d t
      1 -> details ++ show1d t
      2 -> details ++ show2d t
      _ -> details
    where
      -- TODO: this is obviously not the right way to do it,
      -- and will be terribly slow, so please fix it.
      showElems elemShow sep t = "[" ++ (intercalate sep $ map elemShow [t @@ i | i <- [0..((size t 0) - 1)]]) ++ "]"
      padPositive x s = if x >= 0 then " " ++ s else s
      -- TODO: this assumes that scientific notation only uses one-digit exponents, which is not
      --       true in general
      padLarge x s = if (abs x) >= 0.1 then s ++ "   " else s
      show0d x = if isIntegral (dtype t)
                  then padPositive (toInt x) $ show $ toInt x
                  else padLarge (toDouble x) $ padPositive (toDouble x) $ showGFloat (Just 4) (toDouble x) ""
      show1d = showElems show0d ", "
      show2d = showElems show1d (",\n " ++ padding)
      details = "Tensor " ++ (show $ dtype t) ++ " " ++ (show $ shape t) ++ " "
      padding = map (const ' ') details


--------------------------------------------------------------------------------

-- Castable instances
--------------------------------------------------------------------------------

-- NB: ATen only defines Castable [ForeignPtr ATen.Tensor] (ForeignPtr ATen.TensorList)
instance Castable [Tensor] (ForeignPtr ATen.TensorList) where
  cast xs f = do
    ptr_list <- mapM (\x -> (cast x return :: IO (ForeignPtr ATen.Tensor))) xs
    cast ptr_list f
  uncast xs f = uncast xs $ \ptr_list -> do
    tensor_list <- mapM (\(x :: ForeignPtr ATen.Tensor) -> uncast x return) ptr_list
    f tensor_list
