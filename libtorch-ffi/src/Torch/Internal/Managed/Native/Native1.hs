
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Managed.Native.Native1 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type
import Torch.Internal.Class
import Torch.Internal.Cast
import Torch.Internal.Objects
import qualified Torch.Internal.Unmanaged.Native.Native1 as Unmanaged


asinh__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
asinh__t = cast1 Unmanaged.asinh__t

asinh_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
asinh_out_tt = cast2 Unmanaged.asinh_out_tt

atanh_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atanh_t = cast1 Unmanaged.atanh_t

atanh__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atanh__t = cast1 Unmanaged.atanh__t

atanh_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atanh_out_tt = cast2 Unmanaged.atanh_out_tt

as_strided_tlll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
as_strided_tlll = cast4 Unmanaged.as_strided_tlll

as_strided_tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
as_strided_tll = cast3 Unmanaged.as_strided_tll

as_strided__tlll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
as_strided__tlll = cast4 Unmanaged.as_strided__tlll

as_strided__tll
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
as_strided__tll = cast3 Unmanaged.as_strided__tll

asin_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
asin_t = cast1 Unmanaged.asin_t

asin__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
asin__t = cast1 Unmanaged.asin__t

asin_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
asin_out_tt = cast2 Unmanaged.asin_out_tt

atan_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atan_t = cast1 Unmanaged.atan_t

atan__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atan__t = cast1 Unmanaged.atan__t

atan_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
atan_out_tt = cast2 Unmanaged.atan_out_tt

baddbmm_tttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
baddbmm_tttss = cast5 Unmanaged.baddbmm_tttss

baddbmm_ttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
baddbmm_ttts = cast4 Unmanaged.baddbmm_ttts

baddbmm_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
baddbmm_ttt = cast3 Unmanaged.baddbmm_ttt

_baddbmm_mkl__tttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
_baddbmm_mkl__tttss = cast5 Unmanaged._baddbmm_mkl__tttss

_baddbmm_mkl__ttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
_baddbmm_mkl__ttts = cast4 Unmanaged._baddbmm_mkl__ttts

_baddbmm_mkl__ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_baddbmm_mkl__ttt = cast3 Unmanaged._baddbmm_mkl__ttt

baddbmm_out_ttttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
baddbmm_out_ttttss = cast6 Unmanaged.baddbmm_out_ttttss

baddbmm_out_tttts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
baddbmm_out_tttts = cast5 Unmanaged.baddbmm_out_tttts

baddbmm_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
baddbmm_out_tttt = cast4 Unmanaged.baddbmm_out_tttt

bartlett_window_lo
  :: Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
bartlett_window_lo = cast2 Unmanaged.bartlett_window_lo

bartlett_window_l
  :: Int64
  -> IO (ForeignPtr Tensor)
bartlett_window_l = cast1 Unmanaged.bartlett_window_l

bartlett_window_lbo
  :: Int64
  -> CBool
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
bartlett_window_lbo = cast3 Unmanaged.bartlett_window_lbo

bartlett_window_lb
  :: Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
bartlett_window_lb = cast2 Unmanaged.bartlett_window_lb

batch_norm_tttttbddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr Tensor)
batch_norm_tttttbddb = cast9 Unmanaged.batch_norm_tttttbddb

quantized_batch_norm_tttttddl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CDouble
  -> CDouble
  -> Int64
  -> IO (ForeignPtr Tensor)
quantized_batch_norm_tttttddl = cast8 Unmanaged.quantized_batch_norm_tttttddl

_batch_norm_impl_index_tttttbddb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CDouble
  -> CDouble
  -> CBool
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Int64)))
_batch_norm_impl_index_tttttbddb = cast9 Unmanaged._batch_norm_impl_index_tttttbddb

_batch_norm_impl_index_backward_ltttttttbdat
  :: Int64
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> CDouble
  -> ForeignPtr (StdArray '(CBool,3))
  -> ForeignPtr Tensor
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_batch_norm_impl_index_backward_ltttttttbdat = cast12 Unmanaged._batch_norm_impl_index_backward_ltttttttbdat

bernoulli_tG
  :: ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
bernoulli_tG = cast2 Unmanaged.bernoulli_tG

bernoulli_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bernoulli_t = cast1 Unmanaged.bernoulli_t

bernoulli_out_ttG
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
bernoulli_out_ttG = cast3 Unmanaged.bernoulli_out_ttG

bernoulli_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bernoulli_out_tt = cast2 Unmanaged.bernoulli_out_tt

bernoulli_tdG
  :: ForeignPtr Tensor
  -> CDouble
  -> ForeignPtr Generator
  -> IO (ForeignPtr Tensor)
bernoulli_tdG = cast3 Unmanaged.bernoulli_tdG

bernoulli_td
  :: ForeignPtr Tensor
  -> CDouble
  -> IO (ForeignPtr Tensor)
bernoulli_td = cast2 Unmanaged.bernoulli_td

bilinear_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bilinear_tttt = cast4 Unmanaged.bilinear_tttt

binary_cross_entropy_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_tttl = cast4 Unmanaged.binary_cross_entropy_tttl

binary_cross_entropy_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_ttt = cast3 Unmanaged.binary_cross_entropy_ttt

binary_cross_entropy_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_tt = cast2 Unmanaged.binary_cross_entropy_tt

binary_cross_entropy_out_ttttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_out_ttttl = cast5 Unmanaged.binary_cross_entropy_out_ttttl

binary_cross_entropy_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_out_tttt = cast4 Unmanaged.binary_cross_entropy_out_tttt

binary_cross_entropy_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_out_ttt = cast3 Unmanaged.binary_cross_entropy_out_ttt

binary_cross_entropy_backward_ttttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_ttttl = cast5 Unmanaged.binary_cross_entropy_backward_ttttl

binary_cross_entropy_backward_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_tttt = cast4 Unmanaged.binary_cross_entropy_backward_tttt

binary_cross_entropy_backward_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_ttt = cast3 Unmanaged.binary_cross_entropy_backward_ttt

binary_cross_entropy_backward_out_tttttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_out_tttttl = cast6 Unmanaged.binary_cross_entropy_backward_out_tttttl

binary_cross_entropy_backward_out_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_out_ttttt = cast5 Unmanaged.binary_cross_entropy_backward_out_ttttt

binary_cross_entropy_backward_out_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_backward_out_tttt = cast4 Unmanaged.binary_cross_entropy_backward_out_tttt

binary_cross_entropy_with_logits_ttttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_ttttl = cast5 Unmanaged.binary_cross_entropy_with_logits_ttttl

binary_cross_entropy_with_logits_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_tttt = cast4 Unmanaged.binary_cross_entropy_with_logits_tttt

binary_cross_entropy_with_logits_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_ttt = cast3 Unmanaged.binary_cross_entropy_with_logits_ttt

binary_cross_entropy_with_logits_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_tt = cast2 Unmanaged.binary_cross_entropy_with_logits_tt

binary_cross_entropy_with_logits_backward_tttttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_backward_tttttl = cast6 Unmanaged.binary_cross_entropy_with_logits_backward_tttttl

binary_cross_entropy_with_logits_backward_ttttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_backward_ttttt = cast5 Unmanaged.binary_cross_entropy_with_logits_backward_ttttt

binary_cross_entropy_with_logits_backward_tttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_backward_tttt = cast4 Unmanaged.binary_cross_entropy_with_logits_backward_tttt

binary_cross_entropy_with_logits_backward_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
binary_cross_entropy_with_logits_backward_ttt = cast3 Unmanaged.binary_cross_entropy_with_logits_backward_ttt

bincount_ttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr Tensor)
bincount_ttl = cast3 Unmanaged.bincount_ttl

bincount_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bincount_tt = cast2 Unmanaged.bincount_tt

bincount_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bincount_t = cast1 Unmanaged.bincount_t

bitwise_not_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_not_t = cast1 Unmanaged.bitwise_not_t

bitwise_not_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bitwise_not_out_tt = cast2 Unmanaged.bitwise_not_out_tt

logical_not_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_not_t = cast1 Unmanaged.logical_not_t

logical_not_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_not_out_tt = cast2 Unmanaged.logical_not_out_tt

logical_xor_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_xor_tt = cast2 Unmanaged.logical_xor_tt

logical_xor_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_xor_out_ttt = cast3 Unmanaged.logical_xor_out_ttt

logical_and_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_and_tt = cast2 Unmanaged.logical_and_tt

logical_and_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_and_out_ttt = cast3 Unmanaged.logical_and_out_ttt

logical_or_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_or_tt = cast2 Unmanaged.logical_or_tt

logical_or_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
logical_or_out_ttt = cast3 Unmanaged.logical_or_out_ttt

blackman_window_lo
  :: Int64
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
blackman_window_lo = cast2 Unmanaged.blackman_window_lo

blackman_window_l
  :: Int64
  -> IO (ForeignPtr Tensor)
blackman_window_l = cast1 Unmanaged.blackman_window_l

blackman_window_lbo
  :: Int64
  -> CBool
  -> ForeignPtr TensorOptions
  -> IO (ForeignPtr Tensor)
blackman_window_lbo = cast3 Unmanaged.blackman_window_lbo

blackman_window_lb
  :: Int64
  -> CBool
  -> IO (ForeignPtr Tensor)
blackman_window_lb = cast2 Unmanaged.blackman_window_lb

bmm_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bmm_tt = cast2 Unmanaged.bmm_tt

_bmm_ttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
_bmm_ttb = cast3 Unmanaged._bmm_ttb

_bmm_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_bmm_tt = cast2 Unmanaged._bmm_tt

bmm_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
bmm_out_ttt = cast3 Unmanaged.bmm_out_ttt

_bmm_out_tttb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> CBool
  -> IO (ForeignPtr Tensor)
_bmm_out_tttb = cast4 Unmanaged._bmm_out_tttb

_bmm_out_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
_bmm_out_ttt = cast3 Unmanaged._bmm_out_ttt

broadcast_tensors_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr TensorList)
broadcast_tensors_l = cast1 Unmanaged.broadcast_tensors_l

cat_ll
  :: ForeignPtr TensorList
  -> Int64
  -> IO (ForeignPtr Tensor)
cat_ll = cast2 Unmanaged.cat_ll

cat_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr Tensor)
cat_l = cast1 Unmanaged.cat_l

cat_out_tll
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> Int64
  -> IO (ForeignPtr Tensor)
cat_out_tll = cast3 Unmanaged.cat_out_tll

cat_out_tl
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> IO (ForeignPtr Tensor)
cat_out_tl = cast2 Unmanaged.cat_out_tl

cat_ln
  :: ForeignPtr TensorList
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
cat_ln = cast2 Unmanaged.cat_ln

cat_out_tln
  :: ForeignPtr Tensor
  -> ForeignPtr TensorList
  -> ForeignPtr Dimname
  -> IO (ForeignPtr Tensor)
cat_out_tln = cast3 Unmanaged.cat_out_tln

block_diag_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr Tensor)
block_diag_l = cast1 Unmanaged.block_diag_l

ceil_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ceil_t = cast1 Unmanaged.ceil_t

ceil__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ceil__t = cast1 Unmanaged.ceil__t

ceil_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
ceil_out_tt = cast2 Unmanaged.ceil_out_tt

chain_matmul_l
  :: ForeignPtr TensorList
  -> IO (ForeignPtr Tensor)
chain_matmul_l = cast1 Unmanaged.chain_matmul_l

chunk_tll
  :: ForeignPtr Tensor
  -> Int64
  -> Int64
  -> IO (ForeignPtr TensorList)
chunk_tll = cast3 Unmanaged.chunk_tll

chunk_tl
  :: ForeignPtr Tensor
  -> Int64
  -> IO (ForeignPtr TensorList)
chunk_tl = cast2 Unmanaged.chunk_tl

clamp_tss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_tss = cast3 Unmanaged.clamp_tss

clamp_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_ts = cast2 Unmanaged.clamp_ts

clamp_t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
clamp_t = cast1 Unmanaged.clamp_t

clamp__tss
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp__tss = cast3 Unmanaged.clamp__tss

clamp__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp__ts = cast2 Unmanaged.clamp__ts

clamp__t
  :: ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
clamp__t = cast1 Unmanaged.clamp__t

clamp_out_ttss
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_out_ttss = cast4 Unmanaged.clamp_out_ttss

clamp_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_out_tts = cast3 Unmanaged.clamp_out_tts

clamp_out_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
clamp_out_tt = cast2 Unmanaged.clamp_out_tt

clamp_max_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_max_ts = cast2 Unmanaged.clamp_max_ts

clamp_max__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_max__ts = cast2 Unmanaged.clamp_max__ts

clamp_max_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_max_out_tts = cast3 Unmanaged.clamp_max_out_tts

clamp_min_ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_min_ts = cast2 Unmanaged.clamp_min_ts

clamp_min__ts
  :: ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_min__ts = cast2 Unmanaged.clamp_min__ts

clamp_min_out_tts
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
clamp_min_out_tts = cast3 Unmanaged.clamp_min_out_tts

cudnn_is_acceptable_t
  :: ForeignPtr Tensor
  -> IO (CBool)
cudnn_is_acceptable_t = cast1 Unmanaged.cudnn_is_acceptable_t

constant_pad_nd_tls
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr Scalar
  -> IO (ForeignPtr Tensor)
constant_pad_nd_tls = cast3 Unmanaged.constant_pad_nd_tls

constant_pad_nd_tl
  :: ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
constant_pad_nd_tl = cast2 Unmanaged.constant_pad_nd_tl

convolution_tttlllbll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
convolution_tttlllbll = cast9 Unmanaged.convolution_tttlllbll

convolution_overrideable_tttlllbll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
convolution_overrideable_tttlllbll = cast9 Unmanaged.convolution_overrideable_tttlllbll

convolution_backward_overrideable_tttlllblla
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> Int64
  -> ForeignPtr (StdArray '(CBool,3))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
convolution_backward_overrideable_tttlllblla = cast10 Unmanaged.convolution_backward_overrideable_tttlllblla

_convolution_tttlllbllbbb
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> CBool
  -> CBool
  -> IO (ForeignPtr Tensor)
_convolution_tttlllbllbbb = cast12 Unmanaged._convolution_tttlllbllbbb

_convolution_nogroup_tttlllbl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
_convolution_nogroup_tttlllbl = cast8 Unmanaged._convolution_nogroup_tttlllbl

_convolution_double_backward_ttttttlllbllbbba
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> CBool
  -> ForeignPtr IntArray
  -> Int64
  -> CBool
  -> CBool
  -> CBool
  -> ForeignPtr (StdArray '(CBool,3))
  -> IO (ForeignPtr (StdTuple '(Tensor,Tensor,Tensor)))
_convolution_double_backward_ttttttlllbllbbba = cast16 Unmanaged._convolution_double_backward_ttttttlllbllbbba

conv1d_tttllll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
conv1d_tttllll = cast7 Unmanaged.conv1d_tttllll

conv1d_tttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv1d_tttlll = cast6 Unmanaged.conv1d_tttlll

conv1d_tttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv1d_tttll = cast5 Unmanaged.conv1d_tttll

conv1d_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv1d_tttl = cast4 Unmanaged.conv1d_tttl

conv1d_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
conv1d_ttt = cast3 Unmanaged.conv1d_ttt

conv1d_tt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
conv1d_tt = cast2 Unmanaged.conv1d_tt

conv2d_tttllll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> Int64
  -> IO (ForeignPtr Tensor)
conv2d_tttllll = cast7 Unmanaged.conv2d_tttllll

conv2d_tttlll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv2d_tttlll = cast6 Unmanaged.conv2d_tttlll

conv2d_tttll
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv2d_tttll = cast5 Unmanaged.conv2d_tttll

conv2d_tttl
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr IntArray
  -> IO (ForeignPtr Tensor)
conv2d_tttl = cast4 Unmanaged.conv2d_tttl

conv2d_ttt
  :: ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> ForeignPtr Tensor
  -> IO (ForeignPtr Tensor)
conv2d_ttt = cast3 Unmanaged.conv2d_ttt

