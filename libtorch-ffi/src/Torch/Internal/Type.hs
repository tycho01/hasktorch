
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module Torch.Internal.Type where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

import Foreign.C.String
import Foreign.C.Types
import Foreign

type ScalarType = Int8
type DeviceType = Int16
type Backend = CInt
type Layout = Int8
type MemoryFormat = Int8
type QScheme = Int8

-- std::vector<a>
data StdVector a

-- std::array<a>
data StdArray a

-- std::tuple<a>
data StdTuple a

-- at::Tensor
data Tensor

-- std::vector<at::Tensor>
type TensorList = StdVector Tensor

data TensorIndex

data Scalar
data TensorOptions

data IntArrayRef
-- std::vector<int64>
type IntArray = StdVector Int64

data Storage

data StdString
data Generator
data Device
data Context

data C10Ptr a
data Quantizer
-- c10::intrusive_ptr<Quantizer>
type ConstQuantizerPtr = C10Ptr Quantizer

data Dimname
type DimnameList = StdVector Dimname

data Symbol

data IValue
type IValueList = StdVector IValue

data C10Dict a
data C10List a
data C10Optional a

data IVNone
data IVObject
data IVTuple
data IVFuture
data IVConstantString
data Capsule
data Blob

data Module
data SharedPtr a
data JitGraph
data JitNode
data JitValue

typeTable = Map.fromList [
        (C.TypeName "std::array", [t|StdArray|])
      , (C.TypeName "std::vector", [t|StdVector|])
      , (C.TypeName "std::tuple", [t|StdTuple|])
      , (C.TypeName "at::Scalar", [t|Scalar|])
      , (C.TypeName "at::Tensor", [t|Tensor|])
      , (C.TypeName "at::TensorOptions", [t|TensorOptions|])
      , (C.TypeName "at::IntArrayRef", [t|IntArrayRef|])
      , (C.TypeName "at::ScalarType", [t|ScalarType|])
      , (C.TypeName "at::DeviceType", [t|DeviceType|])
      , (C.TypeName "at::Storage", [t|Storage|])
      , (C.TypeName "c10::Device", [t|Device|])
      , (C.TypeName "at::Generator", [t|Generator|])
      , (C.TypeName "std::string", [t|StdString|])
      , (C.TypeName "at::Backend", [t|Backend|])
      , (C.TypeName "at::Layout", [t|Layout|])
      , (C.TypeName "at::MemoryFormat", [t|MemoryFormat|])
      , (C.TypeName "at::Context", [t|Context|])
      , (C.TypeName "at::QScheme", [t|QScheme|])
      , (C.TypeName "at::Dimname", [t|Dimname|])
      , (C.TypeName "at::Symbol", [t|Symbol|])
      , (C.TypeName "Quantizer", [t|Quantizer|])
      , (C.TypeName "at::IValue", [t|IValue|])
      , (C.TypeName "c10::intrusive_ptr", [t|C10Ptr|])
      , (C.TypeName "c10::Dict", [t|C10Dict|])
      , (C.TypeName "c10::List", [t|C10List|])
      , (C.TypeName "c10::optional", [t|C10Optional|])
      , (C.TypeName "at::ivalue::Tuple", [t|IVTuple|])
      , (C.TypeName "at::ivalue::Future", [t|IVFuture|])
      , (C.TypeName "at::ivalue::ConstantString", [t|IVConstantString|])
      , (C.TypeName "at::ivalue::Object", [t|IVObject|])
      , (C.TypeName "torch::jit::CustomClassHolder", [t|Capsule|])
      , (C.TypeName "caffe2::Blob", [t|Blob|])
      , (C.TypeName "torch::jit::script::Module", [t|Module|])
      , (C.TypeName "std::shared_ptr", [t|SharedPtr|])
      , (C.TypeName "torch::jit::Graph", [t|JitGraph|])
      , (C.TypeName "torch::jit::Node", [t|JitNode|])
      , (C.TypeName "torch::jit::Value", [t|JitValue|])
      , (C.TypeName "at::indexing::TensorIndex", [t|TensorIndex|])
    ]
