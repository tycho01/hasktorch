
-- generated by using spec/Declarations.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module ATen.Type where

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

data Tensor
data Scalar
data TensorOptions
data TensorList
data IntArrayRef
data IntArray
data TensorAVector
data SparseTensorRef
data Storage
data StdArray a b
data StdString
data Generator
data Device
data Context

typeTable = Map.fromList [
        (C.TypeName "at::Scalar", [t|Scalar|])
      , (C.TypeName "at::Tensor", [t|Tensor|])
      , (C.TypeName "at::TensorOptions", [t|TensorOptions|])
      , (C.TypeName "std::vector<at::Tensor>", [t|TensorList|])
      , (C.TypeName "at::IntArrayRef", [t|IntArrayRef|])
      , (C.TypeName "std::vector<int64_t>", [t|IntArray|])
      , (C.TypeName "at::ScalarType", [t|ScalarType|])
      , (C.TypeName "at::DeviceType", [t|DeviceType|])
      , (C.TypeName "at::SparseTensorRef", [t|SparseTensorRef|])
      , (C.TypeName "at::Storage", [t|Storage|])
      , (C.TypeName "at::Device", [t|Device|])
      , (C.TypeName "at::Generator", [t|Generator|])
      , (C.TypeName "std::string", [t|StdString|])
      , (C.TypeName "std::array<bool,2>", [t|StdArray CBool 2|])
      , (C.TypeName "std::array<bool,3>", [t|StdArray CBool 3|])
      , (C.TypeName "std::array<bool,4>", [t|StdArray CBool 4|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor>", [t|(Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", [t|(Tensor,Tensor,Tensor,Tensor,Tensor)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>", [t|(Tensor,Tensor,Tensor,TensorList)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,double,int64_t>", [t|(Tensor,Tensor,CDouble,Int64)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,float,int>", [t|(Tensor,Tensor,CFloat,CInt)|])
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>", [t|(Tensor,Tensor,Tensor,Int64)|])
      , (C.TypeName "at::Backend", [t|Backend|])
      , (C.TypeName "at::Layout", [t|Layout|])
      , (C.TypeName "at::Context", [t|Context|])
    ]
