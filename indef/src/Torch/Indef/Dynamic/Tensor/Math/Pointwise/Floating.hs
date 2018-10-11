-------------------------------------------------------------------------------
-- |
-- Module    :  Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating
-- Copyright :  (c) Sam Stites 2017
-- License   :  BSD3
-- Maintainer:  sam@stites.io
-- Stability :  experimental
-- Portability: non-portable
-------------------------------------------------------------------------------
{-# OPTIONS_GHC -fno-cse #-}
module Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating
  ( cinv_    , cinv
  , sigmoid_ , sigmoid
  , log_     , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.log
  , lgamma_  , lgamma
  , log1p_   , log1p
  , exp_     , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.exp
  , cos_     , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.cos
  , acos_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.acos
  , cosh_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.cosh
  , sin_     , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.sin
  , asin_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.asin
  , sinh_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.sinh
  , tan_     , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.tan
  , atan_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.atan
  , tanh_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.tanh
  , erf_     , erf
  , erfinv_  , erfinv
  , pow_     , pow
  , tpow_    , tpow
  , sqrt_    , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.sqrt
  , rsqrt_   , rsqrt
  , ceil_    , ceil
  , floor_   , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.floor
  , round_   , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.round
  , trunc_   , trunc
  , frac_    , frac
  , lerp_    , lerp
  , atan2_   , Torch.Indef.Dynamic.Tensor.Math.Pointwise.Floating.atan2
  ) where

import GHC.Int
import System.IO.Unsafe
import Torch.Indef.Dynamic.Tensor
import qualified Torch.Sig.Tensor.Math.Pointwise.Floating as Sig

import Torch.Indef.Types

-- | Return a new tensor applying @1.0 / x@ to all elements.
cinv :: Dynamic -> Dynamic
cinv t = unsafeDupablePerformIO $ let r = empty in _cinv r t >> pure r
{-# NOINLINE cinv #-}

-- | Inplace version of 'cinv', mutating the first tensor argument.
cinv_ :: Dynamic -> IO ()
cinv_ t = _cinv t t

-- | Returns a new Tensor with the sigmoid of the elements of x.
sigmoid :: Dynamic -> Dynamic
sigmoid t = unsafeDupablePerformIO $ let r = empty in _sigmoid r t >> pure r
{-# NOINLINE sigmoid #-}

-- | Inplace version of 'sigmoid', mutating the first tensor argument.
sigmoid_ :: Dynamic -> IO ()
sigmoid_ t = _sigmoid t t

-- | Returns a new tensor with the natural logarithm of the elements of x.
log :: Dynamic -> Dynamic
log t = unsafeDupablePerformIO $ let r = empty in _log r t >> pure r
{-# NOINLINE log #-}

-- | Inplace version of 'log', mutating the first tensor argument.
log_ :: Dynamic -> IO ()
log_ t = _log t t

-- | Returns a new tensor with the natural logarithm of the absolute value of the gamma function of the elements of x.
lgamma :: Dynamic -> Dynamic
lgamma t = unsafeDupablePerformIO $ let r = empty in _lgamma r t >> pure r
{-# NOINLINE lgamma #-}

-- | Inplace version of 'lgamma', mutating the first tensor argument.
lgamma_ :: Dynamic -> IO ()
lgamma_ t = _lgamma t t

-- | Returns a new tensor with the natural logarithm of the elements of x + 1.
--
-- This function is more accurate than log for small values of x.
log1p :: Dynamic -> Dynamic
log1p t = unsafeDupablePerformIO $ let r = empty in _log1p r t >> pure r
{-# NOINLINE log1p #-}

-- | Inplace version of 'log1p', mutating the first tensor argument.
log1p_ :: Dynamic -> IO ()
log1p_ t = _log1p t t

-- | Returns, for each element in x, e (Neper number, the base of
-- natural logarithms) raised to the power of the element in x.
exp :: Dynamic -> Dynamic
exp t = unsafeDupablePerformIO $ let r = empty in _exp r t >> pure r
{-# NOINLINE exp #-}

-- | Inplace version of 'exp', mutating the first tensor argument.
exp_ :: Dynamic -> IO ()
exp_ t = _exp t t

-- | Returns a new tensor with the cosine of the elements of x.
cos :: Dynamic -> Dynamic
cos t = unsafeDupablePerformIO $ let r = empty in _cos r t >> pure r
{-# NOINLINE cos #-}

-- | Inplace version of 'cos', mutating the first tensor argument.
cos_ :: Dynamic -> IO ()
cos_ t = _cos t t

-- | Returns a new tensor with the arcosine of the elements of x.
acos :: Dynamic -> Dynamic
acos t = unsafeDupablePerformIO $ let r = empty in _acos r t >> pure r
{-# NOINLINE acos #-}

-- | Inplace version of 'acos', mutating the first tensor argument.
acos_ :: Dynamic -> IO ()
acos_ t = _acos t t

-- | Returns a new tensor with the hyberbolic cosine of the elements of x.
cosh :: Dynamic -> Dynamic
cosh t = unsafeDupablePerformIO $ let r = empty in _cosh r t >> pure r
{-# NOINLINE cosh #-}

-- | Inplace version of 'cosh', mutating the first tensor argument.
cosh_ :: Dynamic -> IO ()
cosh_ t = _cosh t t

-- | Returns a new tensor with the sine of the elements of x.
sin :: Dynamic -> Dynamic
sin t = unsafeDupablePerformIO $ let r = empty in _sin r t >> pure r
{-# NOINLINE sin #-}

-- | Inplace version of 'sin', mutating the first tensor argument.
sin_ :: Dynamic -> IO ()
sin_ t = _sin t t

-- | Returns a new tensor with the arcsine of the elements of x.
asin :: Dynamic -> Dynamic
asin t = unsafeDupablePerformIO $ let r = empty in _asin r t >> pure r
{-# NOINLINE asin #-}

-- | Inplace version of 'asin', mutating the first tensor argument.
asin_ :: Dynamic -> IO ()
asin_ t = _asin t t

-- | Returns a new tensor with the hyperbolic sine of the elements of x.
sinh :: Dynamic -> Dynamic
sinh t = unsafeDupablePerformIO $ let r = empty in _sinh r t >> pure r
{-# NOINLINE sinh #-}

-- | Inplace version of 'sinh', mutating the first tensor argument.
sinh_ :: Dynamic -> IO ()
sinh_ t = _sinh t t

-- | Returns a new tensor with the tangent of the elements of x.
tan :: Dynamic -> Dynamic
tan t = unsafeDupablePerformIO $ let r = empty in _tan r t >> pure r
{-# NOINLINE tan #-}

-- | Inplace version of 'tan', mutating the first tensor argument.
tan_ :: Dynamic -> IO ()
tan_ t = _tan t t

-- | Returns a new tensor with the arctangent of the elements of x.
atan :: Dynamic -> Dynamic
atan t = unsafeDupablePerformIO $ let r = empty in _atan r t >> pure r
{-# NOINLINE atan #-}

-- | Inplace version of 'atan', mutating the first tensor argument.
atan_ :: Dynamic -> IO ()
atan_ t = _atan t t

-- | Returns a new tensor with the hyperbolic tangent of the elements of x.
tanh :: Dynamic -> Dynamic
tanh t = unsafeDupablePerformIO $ let r = empty in _tanh r t >> pure r
{-# NOINLINE tanh #-}

-- | Inplace version of 'tanh', mutating the first tensor argument.
tanh_ :: Dynamic -> IO ()
tanh_ t = _tanh t t

-- | Returns a new tensor with the gauss error function applied to the elements of x.
--
-- The error function comes from https://en.wikipedia.org/wiki/Error_function
erf :: Dynamic -> Dynamic
erf t = unsafeDupablePerformIO $ let r = empty in _erf r t >> pure r
{-# NOINLINE erf #-}

-- | Inplace version of 'erf', mutating the first tensor argument.
erf_ :: Dynamic -> IO ()
erf_ t = _erf t t

-- | Returns a new tensor with the inverse gauss error function applied to the elements of x.
--
-- See https://en.wikipedia.org/wiki/Error_function for the gauss error function. This is its inverse.
erfinv :: Dynamic -> Dynamic
erfinv t = unsafeDupablePerformIO $ let r = empty in _erfinv r t >> pure r
{-# NOINLINE erfinv #-}

-- | Inplace version of 'erfinv', mutating the first tensor argument.
erfinv_ :: Dynamic -> IO ()
erfinv_ t = _erfinv t t

-- | Returns a new tensor with the elements of @x@ to the power of @n@.
pow :: Dynamic -> HsReal -> Dynamic
pow t v = unsafeDupablePerformIO $ let r = empty in _pow r t v >> pure r
{-# NOINLINE pow #-}

-- | Inplace version of 'pow', mutating the first tensor argument.
pow_ :: Dynamic -> HsReal -> IO ()
pow_ t v = _pow t t v

-- | Returns a new tensor with the scalar @n@, raised to the power of each element in the tensor @x@.
tpow
  :: HsReal      -- ^ base scalar @n@
  -> Dynamic     -- ^ tensor @x@ of powers to raise @n@ by.
  -> Dynamic
tpow v t = unsafeDupablePerformIO $ let r = empty in _tpow r v t >> pure r
{-# NOINLINE tpow #-}

-- | Inplace version of 'tpow', mutating the first tensor argument.
tpow_ :: HsReal -> Dynamic -> IO ()
tpow_ v t = _tpow t v t

-- | Returns a new tensor with the square root of the elements of x.
sqrt :: Dynamic -> Dynamic
sqrt t = unsafeDupablePerformIO $ let r = empty in _sqrt r t >> pure r
{-# NOINLINE sqrt #-}

-- | Inplace version of 'sqrt', mutating the first tensor argument.
sqrt_ :: Dynamic -> IO ()
sqrt_ t = _sqrt t t

-- | Returns a new tensor with the reciprocal of the square root of the elements of x.
rsqrt :: Dynamic -> Dynamic
rsqrt t = unsafeDupablePerformIO $ let r = empty in _rsqrt r t >> pure r
{-# NOINLINE rsqrt #-}

-- | Inplace version of 'rsqrt', mutating the first tensor argument.
rsqrt_ :: Dynamic -> IO ()
rsqrt_ t = _rsqrt t t

-- | Returns a new tensor with the values of the elements of x
-- rounded up to the nearest integers.
ceil :: Dynamic -> Dynamic
ceil t = unsafeDupablePerformIO $ let r = empty in _ceil r t >> pure r
{-# NOINLINE ceil #-}

-- | Inplace version of 'ceil', mutating the first tensor argument.
ceil_ :: Dynamic -> IO ()
ceil_ t = _ceil t t

-- | Returns a new Tensor with the values of the elements of x
-- rounded down to the nearest integers.
floor :: Dynamic -> Dynamic
floor t = unsafeDupablePerformIO $ let r = empty in _floor r t >> pure r
{-# NOINLINE floor #-}

-- | Inplace version of 'floor', mutating the first tensor argument.
floor_ :: Dynamic -> IO ()
floor_ t = _floor t t

-- | Returns a new tensor with the values of the elements of x
-- rounded to the nearest integers.
--
-- FIXME: The lua docs don't state how this rounding works. Someone
-- should read the source code and document this.
round :: Dynamic -> Dynamic
round t = unsafeDupablePerformIO $ let r = empty in _round r t >> pure r
{-# NOINLINE round #-}

-- | Inplace version of 'round', mutating the first tensor argument.
round_ :: Dynamic -> IO ()
round_ t = _round t t

-- | Returns a new tensor with the truncated integer values of the
-- elements of x.
--
-- FIXME: The lua docs don't state how this truncation works. Someone
-- should read the source code, document this, and explain how this
-- differs from 'floor'.
trunc :: Dynamic -> Dynamic
trunc t = unsafeDupablePerformIO $ let r = empty in _trunc r t >> pure r
{-# NOINLINE trunc #-}

-- | Inplace version of 'trunc', mutating the first tensor argument.
trunc_ :: Dynamic -> IO ()
trunc_ t = _trunc t t

-- | Returns a new tensor with the fractional portion of the elements
-- of x.
frac :: Dynamic -> Dynamic
frac t = unsafeDupablePerformIO $ let r = empty in _frac r t >> pure r
{-# NOINLINE frac #-}

-- | Inplace version of 'frac', mutating the first tensor argument.
frac_ :: Dynamic -> IO ()
frac_ t = _frac t t

-- | Linear interpolation of two scalars or tensors based on a weight:
--
-- @
--   res = a + weight * (b - a)
-- @
lerp :: Dynamic -> Dynamic -> HsReal -> Dynamic
lerp a b v = unsafeDupablePerformIO $ let r = empty in _lerp r a b v >> pure r
{-# NOINLINE lerp #-}

-- | Inplace version of 'lerp', mutating the first tensor argument.
lerp_ :: Dynamic -> Dynamic -> HsReal -> IO ()
lerp_ a b v = _lerp a a b v

-- | Returns a new tensor with the arctangent of the elements
-- of x and y. Note that the arctangent of the elements x and y
-- refers to the signed angle in radians between the rays ending
-- at origin where the first one starts at (1, 0) and the second
-- at (y, x).
atan2 :: Dynamic -> Dynamic -> Dynamic
atan2 a b = unsafeDupablePerformIO $ let r = empty in _atan2 r a b >> pure r
{-# NOINLINE atan2 #-}

-- | Inplace version of 'atan2', mutating the first tensor argument.
atan2_ :: Dynamic -> Dynamic -> IO ()
atan2_ a b = _atan2 a a b


-- class CPUTensorMathPointwiseFloating t where
--   histc_        :: Dynamic -> Dynamic -> Int64 -> HsReal -> HsReal -> IO ()
--   bhistc_       :: Dynamic -> Dynamic -> Int64 -> HsReal -> HsReal -> IO ()


-- ========================================================================= --
-- C-style functions

_cinv, _sigmoid, _log, _lgamma, _log1p, _exp, _cos, _acos, _sinh
  :: Dynamic -> Dynamic -> IO ()
_cinv    a b = with2DynamicState a b Sig.c_cinv
_sigmoid a b = with2DynamicState a b Sig.c_sigmoid
_log     a b = with2DynamicState a b Sig.c_log
_lgamma  a b = with2DynamicState a b Sig.c_lgamma
_log1p   a b = with2DynamicState a b Sig.c_log1p
_exp     a b = with2DynamicState a b Sig.c_exp
_cos     a b = with2DynamicState a b Sig.c_cos
_acos    a b = with2DynamicState a b Sig.c_acos
_sinh    a b = with2DynamicState a b Sig.c_sinh

_asin, _cosh, _sin, _tan, _atan, _tanh, _erf, _erfinv, _sqrt
  :: Dynamic -> Dynamic -> IO ()
_asin   a b = with2DynamicState a b Sig.c_asin
_cosh    a b = with2DynamicState a b Sig.c_cosh
_sin     a b = with2DynamicState a b Sig.c_sin
_tan    a b = with2DynamicState a b Sig.c_tan
_atan   a b = with2DynamicState a b Sig.c_atan
_tanh   a b = with2DynamicState a b Sig.c_tanh
_erf    a b = with2DynamicState a b Sig.c_erf
_erfinv a b = with2DynamicState a b Sig.c_erfinv
_sqrt   a b = with2DynamicState a b Sig.c_sqrt

_rsqrt, _ceil, _floor, _round, _trunc, _frac
  :: Dynamic -> Dynamic -> IO ()
_rsqrt a b = with2DynamicState a b Sig.c_rsqrt
_ceil  a b = with2DynamicState a b Sig.c_ceil
_floor a b = with2DynamicState a b Sig.c_floor
_round a b = with2DynamicState a b Sig.c_round
_trunc a b = with2DynamicState a b Sig.c_trunc
_frac  a b = with2DynamicState a b Sig.c_frac

_atan2 :: Dynamic -> Dynamic -> Dynamic -> IO ()
_atan2 a b c = with3DynamicState a b c Sig.c_atan2

_pow :: Dynamic -> Dynamic -> HsReal -> IO ()
_pow a b v = withLift $ Sig.c_pow
  <$> managedState
  <*> managedTensor a
  <*> managedTensor b
  <*> pure (hs2cReal v)

_tpow :: Dynamic -> HsReal -> Dynamic -> IO ()
_tpow a v b = with2DynamicState a b $ \s' a' b' -> Sig.c_tpow s' a' (hs2cReal v) b'

_lerp :: Dynamic -> Dynamic -> Dynamic -> HsReal -> IO ()
_lerp a b c v = with3DynamicState a b c $ \s' a' b' c' -> Sig.c_lerp s' a' b' c' (hs2cReal v)


