{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
module Torch.Class.C.Tensor.Math where

import THTypes
import Foreign
import Foreign.C.Types
import GHC.TypeLits (Nat)
import Torch.Core.Tensor.Dim
import Torch.Class.C.Internal
import GHC.Int
import Torch.Class.C.IsTensor (IsTensor(empty), inplace)
import THRandomTypes (Generator)
import qualified THByteTypes   as B
import qualified THLongTypes   as L

constant :: (IsTensor t, TensorMath t) => Dim (d::[Nat]) -> HsReal t -> IO t
constant d v = inplace (`fill_` v) d

class TensorMath t where
  fill_        :: t -> HsReal t -> IO ()
  zero_        :: t -> IO ()
  maskedFill_  :: t -> B.DynTensor -> HsReal t -> IO ()
  maskedCopy_  :: t -> B.DynTensor -> t -> IO ()
  maskedSelect_ :: t -> t -> B.DynTensor -> IO ()
  nonzero_     :: L.DynTensor -> t -> IO ()
  indexSelect_ :: t -> t -> Int32 -> L.DynTensor -> IO ()
  indexCopy_   :: t -> Int32 -> L.DynTensor -> t -> IO ()
  indexAdd_    :: t -> Int32 -> L.DynTensor -> t -> IO ()
  indexFill_   :: t -> Int32 -> L.DynTensor -> HsReal t -> IO ()
  take_        :: t -> t -> L.DynTensor -> IO ()
  put_         :: t -> L.DynTensor -> t -> Int32 -> IO ()
  gather_      :: t -> t -> Int32 -> L.DynTensor -> IO ()
  scatter_     :: t -> Int32 -> L.DynTensor -> t -> IO ()
  scatterAdd_  :: t -> Int32 -> L.DynTensor -> t -> IO ()
  scatterFill_ :: t -> Int32 -> L.DynTensor -> HsReal t -> IO ()
  dot          :: t -> t -> IO (HsAccReal t)
  minall       :: t -> IO (HsReal t)
  maxall       :: t -> IO (HsReal t)
  medianall    :: t -> IO (HsReal t)
  sumall       :: t -> IO (HsAccReal t)
  prodall      :: t -> IO (HsAccReal t)
  add_         :: t -> t -> HsReal t -> IO ()
  sub_         :: t -> t -> HsReal t -> IO ()
  add_scaled_  :: t -> t -> HsReal t -> HsReal t -> IO ()
  sub_scaled_  :: t -> t -> HsReal t -> HsReal t -> IO ()
  mul_         :: t -> t -> HsReal t -> IO ()
  div_         :: t -> t -> HsReal t -> IO ()
  lshift_      :: t -> t -> HsReal t -> IO ()
  rshift_      :: t -> t -> HsReal t -> IO ()
  fmod_        :: t -> t -> HsReal t -> IO ()
  remainder_   :: t -> t -> HsReal t -> IO ()
  clamp_       :: t -> t -> HsReal t -> HsReal t -> IO ()
  bitand_      :: t -> t -> HsReal t -> IO ()
  bitor_       :: t -> t -> HsReal t -> IO ()
  bitxor_      :: t -> t -> HsReal t -> IO ()
  cadd_        :: t -> t -> HsReal t -> t -> IO ()
  csub_        :: t -> t -> HsReal t -> t -> IO ()
  cmul_        :: t -> t -> t -> IO ()
  cpow_        :: t -> t -> t -> IO ()
  cdiv_        :: t -> t -> t -> IO ()
  clshift_     :: t -> t -> t -> IO ()
  crshift_     :: t -> t -> t -> IO ()
  cfmod_       :: t -> t -> t -> IO ()
  cremainder_  :: t -> t -> t -> IO ()
  cbitand_     :: t -> t -> t -> IO ()
  cbitor_      :: t -> t -> t -> IO ()
  cbitxor_     :: t -> t -> t -> IO ()
  addcmul_     :: t -> t -> HsReal t -> t -> t -> IO ()
  addcdiv_     :: t -> t -> HsReal t -> t -> t -> IO ()
  addmv_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addmm_       :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addr_        :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  addbmm_      :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  baddbmm_     :: t -> HsReal t -> t -> HsReal t -> t -> t -> IO ()
  match_       :: t -> t -> t -> HsReal t -> IO ()
  numel        :: t -> IO Int64
  max_         :: (t, L.DynTensor) -> t -> Int32 -> Int32 -> IO ()
  min_         :: (t, L.DynTensor) -> t -> Int32 -> Int32 -> IO ()
  kthvalue_    :: (t, L.DynTensor) -> t -> Int64 -> Int32 -> Int32 -> IO ()
  mode_        :: (t, L.DynTensor) -> t -> Int32 -> Int32 -> IO ()
  median_      :: (t, L.DynTensor) -> t -> Int32 -> Int32 -> IO ()
  sum_         :: t -> t -> Int32 -> Int32 -> IO ()
  prod_        :: t -> t -> Int32 -> Int32 -> IO ()
  cumsum_      :: t -> t -> Int32 -> IO ()
  cumprod_     :: t -> t -> Int32 -> IO ()
  sign_        :: t -> t -> IO ()
  trace        :: t -> IO (HsAccReal t)
  cross_       :: t -> t -> t -> Int32 -> IO ()
  cmax_        :: t -> t -> t -> IO ()
  cmin_        :: t -> t -> t -> IO ()
  cmaxValue_   :: t -> t -> HsReal t -> IO ()
  cminValue_   :: t -> t -> HsReal t -> IO ()
  zeros_       :: t -> L.Storage -> IO ()
  zerosLike_   :: t -> t -> IO ()
  ones_        :: t -> L.Storage -> IO ()
  onesLike_    :: t -> t -> IO ()
  diag_        :: t -> t -> Int32 -> IO ()
  eye_         :: t -> Int64 -> Int64 -> IO ()
  arange_      :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  range_       :: t -> HsAccReal t-> HsAccReal t-> HsAccReal t-> IO ()
  randperm_    :: t -> Generator -> Int64 -> IO ()
  reshape_     :: t -> t -> L.Storage -> IO ()
  sort_        :: t -> L.DynTensor -> t -> Int32 -> Int32 -> IO ()
  topk_        :: t -> L.DynTensor -> t -> Int64 -> Int32 -> Int32 -> Int32 -> IO ()
  tril_        :: t -> t -> Int64 -> IO ()
  triu_        :: t -> t -> Int64 -> IO ()
  cat_         :: t -> t -> t -> Int32 -> IO ()
  catArray_    :: t -> [t] -> Int32 -> Int32 -> IO ()
  equal        :: t -> t -> IO Int32
  ltValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  leValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  gtValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  geValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  neValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  eqValue_     :: B.DynTensor -> t -> HsReal t -> IO ()
  ltValueT_    :: t -> t -> HsReal t -> IO ()
  leValueT_    :: t -> t -> HsReal t -> IO ()
  gtValueT_    :: t -> t -> HsReal t -> IO ()
  geValueT_    :: t -> t -> HsReal t -> IO ()
  neValueT_    :: t -> t -> HsReal t -> IO ()
  eqValueT_    :: t -> t -> HsReal t -> IO ()
  ltTensor_    :: B.DynTensor -> t -> t -> IO ()
  leTensor_    :: B.DynTensor -> t -> t -> IO ()
  gtTensor_    :: B.DynTensor -> t -> t -> IO ()
  geTensor_    :: B.DynTensor -> t -> t -> IO ()
  neTensor_    :: B.DynTensor -> t -> t -> IO ()
  eqTensor_    :: B.DynTensor -> t -> t -> IO ()
  ltTensorT_   :: t -> t -> t -> IO ()
  leTensorT_   :: t -> t -> t -> IO ()
  gtTensorT_   :: t -> t -> t -> IO ()
  geTensorT_   :: t -> t -> t -> IO ()
  neTensorT_   :: t -> t -> t -> IO ()
  eqTensorT_   :: t -> t -> t -> IO ()

neg :: (IsTensor t, TensorMathSigned t) => Dim (d::[Nat]) -> t -> IO t
neg d t = inplace (`neg_` t) d

abs :: (IsTensor t, TensorMathSigned t) => Dim (d::[Nat]) -> t -> IO t
abs d t = inplace (`abs_` t) d

class TensorMathSigned t where
  neg_         :: t -> t -> IO ()
  abs_         :: t -> t -> IO ()

class TensorMathFloating t where
  cinv_         :: t -> t -> IO ()
  sigmoid_      :: t -> t -> IO ()
  log_          :: t -> t -> IO ()
  lgamma_       :: t -> t -> IO ()
  log1p_        :: t -> t -> IO ()
  exp_          :: t -> t -> IO ()
  cos_          :: t -> t -> IO ()
  acos_         :: t -> t -> IO ()
  cosh_         :: t -> t -> IO ()
  sin_          :: t -> t -> IO ()
  asin_         :: t -> t -> IO ()
  sinh_         :: t -> t -> IO ()
  tan_          :: t -> t -> IO ()
  atan_         :: t -> t -> IO ()
  atan2_        :: t -> t -> t -> IO ()
  tanh_         :: t -> t -> IO ()
  erf_          :: t -> t -> IO ()
  erfinv_       :: t -> t -> IO ()
  pow_          :: t -> t -> HsReal t -> IO ()
  tpow_         :: t -> HsReal t -> t -> IO ()
  sqrt_         :: t -> t -> IO ()
  rsqrt_        :: t -> t -> IO ()
  ceil_         :: t -> t -> IO ()
  floor_        :: t -> t -> IO ()
  round_        :: t -> t -> IO ()
  trunc_        :: t -> t -> IO ()
  frac_         :: t -> t -> IO ()
  lerp_         :: t -> t -> t -> HsReal t -> IO ()
  mean_         :: t -> t -> Int32 -> Int32 -> IO ()
  std_          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  var_          :: t -> t -> Int32 -> Int32 -> Int32 -> IO ()
  norm_         :: t -> t -> HsReal t -> Int32 -> Int32 -> IO ()
  renorm_       :: t -> t -> HsReal t -> Int32 -> HsReal t -> IO ()
  dist          :: t -> t -> HsReal t -> IO (HsAccReal t)
  histc_        :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  bhistc_       :: t -> t -> Int64 -> HsReal t -> HsReal t -> IO ()
  meanall       :: t -> IO (HsAccReal t)
  varall        :: t -> Int32 -> IO (HsAccReal t)
  stdall        :: t -> Int32 -> IO (HsAccReal t)
  normall       :: t -> HsReal t -> IO (HsAccReal t)
  linspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  logspace_     :: t -> HsReal t -> HsReal t -> Int64 -> IO ()
  rand_         :: t -> Generator -> L.Storage -> IO ()
  randn_        :: t -> Generator -> L.Storage -> IO ()

