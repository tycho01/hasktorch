
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.SparseTensorRef where


import qualified Language.C.Inline.Cpp as C
import qualified Language.C.Inline.Cpp.Exceptions as C
import qualified Language.C.Inline.Context as C
import qualified Language.C.Types as C
import qualified Data.Map as Map
import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import Aten.Unmanaged.Type.Tensor

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"



newSparseTensorRef
  :: Ptr Tensor
  -> IO (Ptr SparseTensorRef)
newSparseTensorRef _x =
  [C.block| at::SparseTensorRef* { return new at::SparseTensorRef(
    *$(at::Tensor* _x));
  }|]



deleteSparseTensorRef :: Ptr SparseTensorRef -> IO ()
deleteSparseTensorRef object = [C.block| void { delete $(at::SparseTensorRef* object);}|]

instance CppObject SparseTensorRef where
  fromPtr ptr = newForeignPtr ptr (deleteSparseTensorRef ptr)



