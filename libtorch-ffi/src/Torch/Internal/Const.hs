
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Const where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

import Foreign.C.String
import Foreign.C.Types
import Foreign
import Torch.Internal.Type

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ScalarType.h>"

kByte :: ScalarType
kByte = [C.pure| int8_t { (int8_t) at::ScalarType::Byte } |]

kChar :: ScalarType
kChar = [C.pure| int8_t { (int8_t) at::ScalarType::Char } |]

kDouble :: ScalarType
kDouble = [C.pure| int8_t { (int8_t) at::ScalarType::Double } |]

kFloat :: ScalarType
kFloat = [C.pure| int8_t { (int8_t) at::ScalarType::Float } |]

kInt :: ScalarType
kInt = [C.pure| int8_t { (int8_t) at::ScalarType::Int } |]

kLong :: ScalarType
kLong = [C.pure| int8_t { (int8_t) at::ScalarType::Long } |]

kShort :: ScalarType
kShort = [C.pure| int8_t { (int8_t) at::ScalarType::Short } |]

kHalf :: ScalarType
kHalf = [C.pure| int8_t { (int8_t) at::ScalarType::Half } |]

kBool :: ScalarType
kBool = [C.pure| int8_t { (int8_t) at::ScalarType::Bool } |]

kComplexHalf :: ScalarType
kComplexHalf = [C.pure| int8_t { (int8_t) at::ScalarType::ComplexHalf } |]

kComplexFloat :: ScalarType
kComplexFloat = [C.pure| int8_t { (int8_t) at::ScalarType::ComplexFloat } |]

kComplexDouble :: ScalarType
kComplexDouble = [C.pure| int8_t { (int8_t) at::ScalarType::ComplexDouble } |]

kUndefined :: ScalarType
kUndefined = [C.pure| int8_t { (int8_t) at::ScalarType::Undefined } |]

kCPU :: DeviceType
kCPU = [C.pure| int16_t { (int16_t) at::DeviceType::CPU } |]

kCUDA :: DeviceType
kCUDA = [C.pure| int16_t { (int16_t) at::DeviceType::CUDA } |]

kMKLDNN :: DeviceType
kMKLDNN = [C.pure| int16_t { (int16_t) at::DeviceType::MKLDNN } |]

kOPENGL :: DeviceType
kOPENGL = [C.pure| int16_t { (int16_t) at::DeviceType::OPENGL } |]

kOPENCL :: DeviceType
kOPENCL = [C.pure| int16_t { (int16_t) at::DeviceType::OPENCL } |]

kIDEEP :: DeviceType
kIDEEP = [C.pure| int16_t { (int16_t) at::DeviceType::IDEEP } |]

kHIP :: DeviceType
kHIP = [C.pure| int16_t { (int16_t) at::DeviceType::HIP } |]

kFPGA :: DeviceType
kFPGA = [C.pure| int16_t { (int16_t) at::DeviceType::FPGA } |]

kMSNPU :: DeviceType
kMSNPU = [C.pure| int16_t { (int16_t) at::DeviceType::MSNPU } |]

kXLA :: DeviceType
kXLA = [C.pure| int16_t { (int16_t) at::DeviceType::XLA } |]

kCOMPILE_TIME_MAX_DEVICE_TYPES :: DeviceType
kCOMPILE_TIME_MAX_DEVICE_TYPES = [C.pure| int16_t { (int16_t) at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES } |]

kONLY_FOR_TEST :: DeviceType
kONLY_FOR_TEST = [C.pure| int16_t { (int16_t) at::DeviceType::ONLY_FOR_TEST } |]

-- TODO: add all values for at::Reduction

kMean :: Int64
kMean = [C.pure| int64_t { (int64_t) at::Reduction::Mean } |]

bCPU :: Backend
bCPU = [C.pure| int { (int) at::Backend::CPU } |]

bCUDA :: Backend
bCUDA = [C.pure| int { (int) at::Backend::CUDA } |]

bHIP :: Backend
bHIP = [C.pure| int { (int) at::Backend::HIP } |]

bSparseCPU :: Backend
bSparseCPU = [C.pure| int { (int) at::Backend::SparseCPU } |]

bSparseCUDA :: Backend
bSparseCUDA = [C.pure| int { (int) at::Backend::SparseCUDA } |]

bSparseHIP :: Backend
bSparseHIP = [C.pure| int { (int) at::Backend::SparseHIP } |]

bMSNPU :: Backend
bMSNPU = [C.pure| int { (int) at::Backend::MSNPU } |]

bXLA :: Backend
bXLA = [C.pure| int { (int) at::Backend::XLA } |]

bUndefined :: Backend
bUndefined = [C.pure| int { (int) at::Backend::Undefined } |]

bNumOptions :: Backend
bNumOptions = [C.pure| int { (int) at::Backend::NumOptions } |]

kStrided :: Layout
kStrided = [C.pure| int8_t { (int8_t) at::kStrided } |]

kSparse :: Layout
kSparse = [C.pure| int8_t { (int8_t) at::kSparse } |]

kMkldnn :: Layout
kMkldnn = [C.pure| int8_t { (int8_t) at::kMkldnn } |]
