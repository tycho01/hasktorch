
-- generated by using spec/tuples.yaml

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Torch.Internal.Unmanaged.Type.Tuple where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Torch.Internal.Type
import Torch.Internal.Class

import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"




-----------------StdTuple '(Tensor,Tensor)---------------------

deleteTensorTensor :: Ptr (StdTuple '(Tensor,Tensor)) -> IO ()
deleteTensorTensor ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensor ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor))) where
  type A (Ptr (StdTuple '(Tensor,Tensor))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)---------------------

deleteTensorTensorTensorTensorTensor :: Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)) -> IO ()
deleteTensorTensorTensorTensorTensor ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorTensorTensorTensor ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get2 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple4 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) where
  type D (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get3 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<3>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple5 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) where
  type E (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get4 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<4>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,Tensor,TensorList)---------------------

deleteTensorTensorTensorTensorList :: Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList)) -> IO ()
deleteTensorTensorTensorTensorList ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,TensorList)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorTensorTensorList ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) = Ptr Tensor
  get2 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* v)));}|]

instance CppTuple4 (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) where
  type D (Ptr (StdTuple '(Tensor,Tensor,Tensor,TensorList))) = Ptr TensorList
  get3 v = [C.throwBlock| std::vector<at::Tensor>* { return new std::vector<at::Tensor>(std::get<3>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,std::vector<at::Tensor>>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,Tensor,Int64)---------------------

deleteTensorTensorTensorInt64 :: Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64)) -> IO ()
deleteTensorTensorTensorInt64 ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Int64)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorTensorInt64 ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) = Ptr Tensor
  get2 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>* v)));}|]

instance CppTuple4 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) where
  type D (Ptr (StdTuple '(Tensor,Tensor,Tensor,Int64))) = Int64
  get3 v = [C.throwBlock| int64_t { return (std::get<3>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,Tensor)---------------------

deleteTensorTensorTensor :: Ptr (StdTuple '(Tensor,Tensor,Tensor)) -> IO ()
deleteTensorTensorTensor ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,at::Tensor>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorTensor ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,Tensor))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,Tensor))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,Tensor))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,Tensor))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,Tensor))) = Ptr Tensor
  get2 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,Tensor,Tensor)---------------------

deleteTensorTensorTensorTensor :: Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor)) -> IO ()
deleteTensorTensorTensorTensor ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,Tensor,Tensor)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorTensorTensor ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get2 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]

instance CppTuple4 (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) where
  type D (Ptr (StdTuple '(Tensor,Tensor,Tensor,Tensor))) = Ptr Tensor
  get3 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<3>(*$(std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>* v)));}|]


-----------------StdTuple '(Tensor,Tensor,CDouble,Int64)---------------------

deleteTensorTensorCDoubleInt64 :: Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64)) -> IO ()
deleteTensorTensorCDoubleInt64 ptr = [C.throwBlock| void { delete $(std::tuple<at::Tensor,at::Tensor,double,int64_t>* ptr); return; }|]

instance CppObject (StdTuple '(Tensor,Tensor,CDouble,Int64)) where
  fromPtr ptr = newForeignPtr ptr (deleteTensorTensorCDoubleInt64 ptr)

instance CppTuple2 (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) where
  type A (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) = Ptr Tensor
  type B (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) = Ptr Tensor
  get0 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<0>(*$(std::tuple<at::Tensor,at::Tensor,double,int64_t>* v)));}|]
  get1 v = [C.throwBlock| at::Tensor* { return new at::Tensor(std::get<1>(*$(std::tuple<at::Tensor,at::Tensor,double,int64_t>* v)));}|]

instance CppTuple3 (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) where
  type C (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) = CDouble
  get2 v = [C.throwBlock| double { return (std::get<2>(*$(std::tuple<at::Tensor,at::Tensor,double,int64_t>* v)));}|]

instance CppTuple4 (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) where
  type D (Ptr (StdTuple '(Tensor,Tensor,CDouble,Int64))) = Int64
  get3 v = [C.throwBlock| int64_t { return (std::get<3>(*$(std::tuple<at::Tensor,at::Tensor,double,int64_t>* v)));}|]

