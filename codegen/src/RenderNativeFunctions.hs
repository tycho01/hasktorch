{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE QuasiQuotes #-}
module RenderNativeFunctions where

import Data.Yaml

import qualified Data.Yaml as Y
import Text.Shakespeare.Text (st)
import Data.Text (Text)
import qualified Data.Text.IO as T
import qualified Data.List as L

import ParseNativeFunctions
import ParseFunctionSig as P
import RenderCommon

renderFunctions :: [NativeFunction'] -> Text
renderFunctions nfs = mconcat $ flip map nfs $ \nf -> mconcat $
  case dispatch' nf of
    Nothing -> [functionToCpp True "at::native::" $ func' nf]
    Just d -> map (\c -> functionToCpp True "at::native::" $ (func' nf){name=c}) (uniqFunctions d)
  where
    uniqFunctions d = L.nub $ concat $
      [ case cpu d of
          Nothing -> []
          Just c -> [c]
      , case gpu d of
          Nothing -> []
          Just c -> [c]
      , case cuda d of
          Nothing -> []
          Just c -> [c]
      , case sparseCPU d of
          Nothing -> []
          Just c -> [c]
      , case sparseCUDA d of
          Nothing -> []
          Just c -> [c]
      ]


decodeAndCodeGen :: String -> IO ()
decodeAndCodeGen fileName = do
  funcs <- Y.decodeFileEither fileName :: IO (Either ParseException [NativeFunction'])
  case funcs of
    Left err' -> print err'
    Right fns -> do
      T.writeFile "ffi/NativeFunctions.hs" [st|
-- generated by using spec/native_functions_modified.yaml and deps/libtorch/include/ATen/NativeFunctions.h

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

module NativeFunctions where

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

import Foreign.C.String
import Foreign.C.Types
import Foreign

data Scalar
data Tensor
data TensorOptions
data TensorList
data IndexTensor
data IntList
data StdArray a b
data ScalarType
data SparseTensorRef

data StdString
data Generator
data Device
data Storage

C.context $ C.cppCtx <> mempty {
    C.ctxTypesTable = Map.fromList [
        (C.TypeName "at::Scalar", #{bra}t|Scalar|#{cket})
      , (C.TypeName "at::Tensor", #{bra}t|Tensor|#{cket})
      , (C.TypeName "at::TensorOptions", #{bra}t|TensorOptions|#{cket})
      , (C.TypeName "at::TensorList", #{bra}t|TensorList|#{cket})
      , (C.TypeName "at::IndexTensor", #{bra}t|IndexTensor|#{cket})
      , (C.TypeName "at::IntArrayRef", #{bra}t|IntList|#{cket})
      , (C.TypeName "at::ScalarType", #{bra}t|ScalarType|#{cket})
      , (C.TypeName "at::SparseTensorRef", #{bra}t|SparseTensorRef|#{cket})
      , (C.TypeName "at::Storage", #{bra}t|Storage|#{cket})
      , (C.TypeName "at::Device", #{bra}t|Device|#{cket})
      , (C.TypeName "at::Generator", #{bra}t|Generator|#{cket})
      , (C.TypeName "std::string", #{bra}t|StdString|#{cket})
      , (C.TypeName "std::array<bool,2>", #{bra}t|StdArray CBool 2|#{cket})
      , (C.TypeName "std::array<bool,3>", #{bra}t|StdArray CBool 3|#{cket})
      , (C.TypeName "std::array<bool,4>", #{bra}t|StdArray CBool 4|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>", #{bra}t|(Tensor,Tensor,Tensor,Tensor,Tensor)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,at::Tensor,at::TensorList>", #{bra}t|(Tensor,Tensor,Tensor,TensorList)|#{cket})
      , (C.TypeName "std::tuple<at::Tensor,at::Tensor,double,int64_t>", #{bra}t|(Tensor,Tensor,CDouble,Int64)|#{cket})
    ]
}

C.include "<ATen/ATen.h>"

#{renderFunctions fns}
|]
