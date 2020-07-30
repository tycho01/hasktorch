
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Unmanaged.Native.Native7 where


import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<vector>"
C.include "<ATen/Tensor.h>"
C.include "<ATen/Functions.h>"


sin_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sin_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin(
    *$(at::Tensor* _self)));
  }|]

sin__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sin__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin_(
    *$(at::Tensor* _self)));
  }|]

sin_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sin_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sin_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

sinh_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sinh_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh(
    *$(at::Tensor* _self)));
  }|]

sinh__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sinh__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh_(
    *$(at::Tensor* _self)));
  }|]

sinh_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sinh_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sinh_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

detach_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
detach_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::detach(
    *$(at::Tensor* _self)));
  }|]

detach__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
detach__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::detach_(
    *$(at::Tensor* _self)));
  }|]

size_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
size_tl _self _dim =
  [C.throwBlock| int64_t { return (at::size(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

size_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
size_tn _self _dim =
  [C.throwBlock| int64_t { return (at::size(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

slice_tllll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tllll _self _dim _start _end _step =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)
  , $(int64_t _step)));
  }|]

slice_tlll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tlll _self _dim _start _end =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)
  , $(int64_t _end)));
  }|]

slice_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
slice_tll _self _dim _start =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(int64_t _start)));
  }|]

slice_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
slice_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

slice_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
slice_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::slice(
    *$(at::Tensor* _self)));
  }|]

slogdet_t
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
slogdet_t _self =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::slogdet(
    *$(at::Tensor* _self)));
  }|]

smm_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
smm_tt _self _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::smm(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _mat2)));
  }|]

softmax_tls
  :: Ptr Tensor
  -> Int64
  -> ScalarType
  -> IO (Ptr Tensor)
softmax_tls _self _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(at::ScalarType _dtype)));
  }|]

softmax_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
softmax_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

softmax_tns
  :: Ptr Tensor
  -> Ptr Dimname
  -> ScalarType
  -> IO (Ptr Tensor)
softmax_tns _self _dim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(at::ScalarType _dtype)));
  }|]

softmax_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
softmax_tn _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::softmax(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

_softmax_tlb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
_softmax_tlb _self _dim _half_to_float =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_softmax(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _half_to_float)));
  }|]

_softmax_backward_data_ttlt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> Ptr Tensor
  -> IO (Ptr Tensor)
_softmax_backward_data_ttlt _grad_output _output _dim _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_softmax_backward_data(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _output)
  , $(int64_t _dim)
  , *$(at::Tensor* _self)));
  }|]

split_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr TensorList)
split_tll _self _split_size _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)
  , $(int64_t _dim)));
  }|]

split_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr TensorList)
split_tl _self _split_size =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split(
    *$(at::Tensor* _self)
  , $(int64_t _split_size)));
  }|]

split_with_sizes_tll
  :: Ptr Tensor
  -> Ptr IntArray
  -> Int64
  -> IO (Ptr TensorList)
split_with_sizes_tll _self _split_sizes _dim =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split_with_sizes(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _split_sizes)
  , $(int64_t _dim)));
  }|]

split_with_sizes_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr TensorList)
split_with_sizes_tl _self _split_sizes =
  [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(at::split_with_sizes(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _split_sizes)));
  }|]

squeeze_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
squeeze_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::squeeze(
    *$(at::Tensor* _self)));
  }|]

squeeze_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
squeeze_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::squeeze(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

squeeze_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
squeeze_tn _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::squeeze(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

sspaddmm_tttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
sspaddmm_tttss _self _mat1 _mat2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

sspaddmm_ttts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
sspaddmm_ttts _self _mat1 _mat2 _beta =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)));
  }|]

sspaddmm_ttt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sspaddmm_ttt _self _mat1 _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)));
  }|]

sspaddmm_out_ttttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
sspaddmm_out_ttttss _out _self _mat1 _mat2 _beta _alpha =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)
  , *$(at::Scalar* _alpha)));
  }|]

sspaddmm_out_tttts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
sspaddmm_out_tttts _out _self _mat1 _mat2 _beta =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)
  , *$(at::Scalar* _beta)));
  }|]

sspaddmm_out_tttt
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sspaddmm_out_tttt _out _self _mat1 _mat2 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sspaddmm_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Tensor* _mat1)
  , *$(at::Tensor* _mat2)));
  }|]

stack_ll
  :: Ptr TensorList
  -> Int64
  -> IO (Ptr Tensor)
stack_ll _tensors _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stack(
    *$(std::vector<at::Tensor>* _tensors)
  , $(int64_t _dim)));
  }|]

stack_l
  :: Ptr TensorList
  -> IO (Ptr Tensor)
stack_l _tensors =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stack(
    *$(std::vector<at::Tensor>* _tensors)));
  }|]

stack_out_tll
  :: Ptr Tensor
  -> Ptr TensorList
  -> Int64
  -> IO (Ptr Tensor)
stack_out_tll _out _tensors _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stack_out(
    *$(at::Tensor* _out)
  , *$(std::vector<at::Tensor>* _tensors)
  , $(int64_t _dim)));
  }|]

stack_out_tl
  :: Ptr Tensor
  -> Ptr TensorList
  -> IO (Ptr Tensor)
stack_out_tl _out _tensors =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stack_out(
    *$(at::Tensor* _out)
  , *$(std::vector<at::Tensor>* _tensors)));
  }|]

stft_tllltbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
stft_tllltbb _self _n_fft _hop_length _win_length _window _normalized _onesided =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _normalized)
  , $(bool _onesided)));
  }|]

stft_tllltb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
stft_tllltb _self _n_fft _hop_length _win_length _window _normalized =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _normalized)));
  }|]

stft_tlllt
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> IO (Ptr Tensor)
stft_tlllt _self _n_fft _hop_length _win_length _window =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)));
  }|]

stft_tlll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
stft_tlll _self _n_fft _hop_length _win_length =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)));
  }|]

stft_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
stft_tll _self _n_fft _hop_length =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)));
  }|]

stft_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
stft_tl _self _n_fft =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::stft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)));
  }|]

istft_tllltbbbl
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> Int64
  -> IO (Ptr Tensor)
istft_tllltbbbl _self _n_fft _hop_length _win_length _window _center _normalized _onesided _length =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)
  , $(bool _normalized)
  , $(bool _onesided)
  , $(int64_t _length)));
  }|]

istft_tllltbbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
istft_tllltbbb _self _n_fft _hop_length _win_length _window _center _normalized _onesided =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)
  , $(bool _normalized)
  , $(bool _onesided)));
  }|]

istft_tllltbb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
istft_tllltbb _self _n_fft _hop_length _win_length _window _center _normalized =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)
  , $(bool _normalized)));
  }|]

istft_tllltb
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
istft_tllltb _self _n_fft _hop_length _win_length _window _center =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)
  , $(bool _center)));
  }|]

istft_tlllt
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> Ptr Tensor
  -> IO (Ptr Tensor)
istft_tlllt _self _n_fft _hop_length _win_length _window =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)
  , *$(at::Tensor* _window)));
  }|]

istft_tlll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
istft_tlll _self _n_fft _hop_length _win_length =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)
  , $(int64_t _win_length)));
  }|]

istft_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
istft_tll _self _n_fft _hop_length =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)
  , $(int64_t _hop_length)));
  }|]

istft_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
istft_tl _self _n_fft =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::istft(
    *$(at::Tensor* _self)
  , $(int64_t _n_fft)));
  }|]

stride_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Int64)
stride_tl _self _dim =
  [C.throwBlock| int64_t { return (at::stride(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

stride_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Int64)
stride_tn _self _dim =
  [C.throwBlock| int64_t { return (at::stride(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

sum_ts
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
sum_ts _self _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , $(at::ScalarType _dtype)));
  }|]

sum_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sum_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)));
  }|]

sum_tlbs
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
sum_tlbs _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

sum_tlb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
sum_tlb _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

sum_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
sum_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)));
  }|]

sum_tNbs
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
sum_tNbs _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

sum_tNb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
sum_tNb _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)));
  }|]

sum_tN
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
sum_tN _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)));
  }|]

sum_out_ttlbs
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
sum_out_ttlbs _out _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

sum_out_ttlb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
sum_out_ttlb _out _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _keepdim)));
  }|]

sum_out_ttl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
sum_out_ttl _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)));
  }|]

sum_out_ttNbs
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
sum_out_ttNbs _out _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

sum_out_ttNb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
sum_out_ttNb _out _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _keepdim)));
  }|]

sum_out_ttN
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
sum_out_ttN _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sum_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)));
  }|]

sqrt_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sqrt_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sqrt(
    *$(at::Tensor* _self)));
  }|]

sqrt__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
sqrt__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sqrt_(
    *$(at::Tensor* _self)));
  }|]

sqrt_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
sqrt_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::sqrt_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

square_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
square_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::square(
    *$(at::Tensor* _self)));
  }|]

square__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
square__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::square_(
    *$(at::Tensor* _self)));
  }|]

std_tb
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr Tensor)
std_tb _self _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , $(bool _unbiased)));
  }|]

std_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
std_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)));
  }|]

std_tlbb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
std_tlbb _self _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_tlb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
std_tlb _self _dim _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)));
  }|]

std_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
std_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)));
  }|]

std_mean_tb
  :: Ptr Tensor
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tb _self _unbiased =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , $(bool _unbiased)));
  }|]

std_mean_t
  :: Ptr Tensor
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_t _self =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)));
  }|]

std_mean_tlbb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tlbb _self _dim _unbiased _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_mean_tlb
  :: Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tlb _self _dim _unbiased =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)));
  }|]

std_mean_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tl _self _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)));
  }|]

std_mean_tNbb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tNbb _self _dim _unbiased _keepdim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_mean_tNb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tNb _self _dim _unbiased =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)));
  }|]

std_mean_tN
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr (StdTuple '(Tensor,Tensor)))
std_mean_tN _self _dim =
  [C.throwBlock| std::tuple<at::Tensor,at::Tensor>* { return new std::tuple<at::Tensor,at::Tensor>(at::std_mean(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)));
  }|]

std_out_ttlbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
std_out_ttlbb _out _self _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_out_ttlb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> CBool
  -> IO (Ptr Tensor)
std_out_ttlb _out _self _dim _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)
  , $(bool _unbiased)));
  }|]

std_out_ttl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
std_out_ttl _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dim)));
  }|]

std_tNbb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
std_tNbb _self _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_tNb
  :: Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
std_tNb _self _dim _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)));
  }|]

std_tN
  :: Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
std_tN _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std(
    *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)));
  }|]

std_out_ttNbb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> CBool
  -> IO (Ptr Tensor)
std_out_ttNbb _out _self _dim _unbiased _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)
  , $(bool _keepdim)));
  }|]

std_out_ttNb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> CBool
  -> IO (Ptr Tensor)
std_out_ttNb _out _self _dim _unbiased =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)
  , $(bool _unbiased)));
  }|]

std_out_ttN
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr DimnameList
  -> IO (Ptr Tensor)
std_out_ttN _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::std_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(std::vector<at::Dimname>* _dim)));
  }|]

prod_ts
  :: Ptr Tensor
  -> ScalarType
  -> IO (Ptr Tensor)
prod_ts _self _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , $(at::ScalarType _dtype)));
  }|]

prod_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
prod_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)));
  }|]

prod_tlbs
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
prod_tlbs _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

prod_tlb
  :: Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
prod_tlb _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

prod_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
prod_tl _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

prod_out_ttlbs
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
prod_out_ttlbs _out _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

prod_out_ttlb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> CBool
  -> IO (Ptr Tensor)
prod_out_ttlb _out _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , $(int64_t _dim)
  , $(bool _keepdim)));
  }|]

prod_out_ttl
  :: Ptr Tensor
  -> Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
prod_out_ttl _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , $(int64_t _dim)));
  }|]

prod_tnbs
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
prod_tnbs _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

prod_tnb
  :: Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
prod_tnb _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

prod_tn
  :: Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
prod_tn _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

prod_out_ttnbs
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> ScalarType
  -> IO (Ptr Tensor)
prod_out_ttnbs _out _self _dim _keepdim _dtype =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(bool _keepdim)
  , $(at::ScalarType _dtype)));
  }|]

prod_out_ttnb
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Dimname
  -> CBool
  -> IO (Ptr Tensor)
prod_out_ttnb _out _self _dim _keepdim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)
  , $(bool _keepdim)));
  }|]

prod_out_ttn
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Dimname
  -> IO (Ptr Tensor)
prod_out_ttn _out _self _dim =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::prod_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Dimname* _dim)));
  }|]

t_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
t_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::t(
    *$(at::Tensor* _self)));
  }|]

tan_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tan_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tan(
    *$(at::Tensor* _self)));
  }|]

tan__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tan__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tan_(
    *$(at::Tensor* _self)));
  }|]

tan_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tan_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tan_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

tanh_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tanh_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tanh(
    *$(at::Tensor* _self)));
  }|]

tanh__t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
tanh__t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tanh_(
    *$(at::Tensor* _self)));
  }|]

tanh_out_tt
  :: Ptr Tensor
  -> Ptr Tensor
  -> IO (Ptr Tensor)
tanh_out_tt _out _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tanh_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)));
  }|]

tensordot_ttll
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr IntArray
  -> Ptr IntArray
  -> IO (Ptr Tensor)
tensordot_ttll _self _other _dims_self _dims_other =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::tensordot(
    *$(at::Tensor* _self)
  , *$(at::Tensor* _other)
  , *$(std::vector<int64_t>* _dims_self)
  , *$(std::vector<int64_t>* _dims_other)));
  }|]

threshold_tss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
threshold_tss _self _threshold _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::threshold(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _threshold)
  , *$(at::Scalar* _value)));
  }|]

threshold__tss
  :: Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
threshold__tss _self _threshold _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::threshold_(
    *$(at::Tensor* _self)
  , *$(at::Scalar* _threshold)
  , *$(at::Scalar* _value)));
  }|]

threshold_out_ttss
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> Ptr Scalar
  -> IO (Ptr Tensor)
threshold_out_ttss _out _self _threshold _value =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::threshold_out(
    *$(at::Tensor* _out)
  , *$(at::Tensor* _self)
  , *$(at::Scalar* _threshold)
  , *$(at::Scalar* _value)));
  }|]

threshold_backward_tts
  :: Ptr Tensor
  -> Ptr Tensor
  -> Ptr Scalar
  -> IO (Ptr Tensor)
threshold_backward_tts _grad_output _self _threshold =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::threshold_backward(
    *$(at::Tensor* _grad_output)
  , *$(at::Tensor* _self)
  , *$(at::Scalar* _threshold)));
  }|]

transpose_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
transpose_tll _self _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::transpose(
    *$(at::Tensor* _self)
  , $(int64_t _dim0)
  , $(int64_t _dim1)));
  }|]

transpose_tnn
  :: Ptr Tensor
  -> Ptr Dimname
  -> Ptr Dimname
  -> IO (Ptr Tensor)
transpose_tnn _self _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::transpose(
    *$(at::Tensor* _self)
  , *$(at::Dimname* _dim0)
  , *$(at::Dimname* _dim1)));
  }|]

_mkldnn_transpose_tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
_mkldnn_transpose_tll _self _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_mkldnn_transpose(
    *$(at::Tensor* _self)
  , $(int64_t _dim0)
  , $(int64_t _dim1)));
  }|]

_mkldnn_transpose__tll
  :: Ptr Tensor
  -> Int64
  -> Int64
  -> IO (Ptr Tensor)
_mkldnn_transpose__tll _self _dim0 _dim1 =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::_mkldnn_transpose_(
    *$(at::Tensor* _self)
  , $(int64_t _dim0)
  , $(int64_t _dim1)));
  }|]

one_hot_tl
  :: Ptr Tensor
  -> Int64
  -> IO (Ptr Tensor)
one_hot_tl _self _num_classes =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::one_hot(
    *$(at::Tensor* _self)
  , $(int64_t _num_classes)));
  }|]

one_hot_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
one_hot_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::one_hot(
    *$(at::Tensor* _self)));
  }|]

flip_tl
  :: Ptr Tensor
  -> Ptr IntArray
  -> IO (Ptr Tensor)
flip_tl _self _dims =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::flip(
    *$(at::Tensor* _self)
  , *$(std::vector<int64_t>* _dims)));
  }|]

fliplr_t
  :: Ptr Tensor
  -> IO (Ptr Tensor)
fliplr_t _self =
  [C.throwBlock| at::Tensor* { return new at::Tensor(at::fliplr(
    *$(at::Tensor* _self)));
  }|]

