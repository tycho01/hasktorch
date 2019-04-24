
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Unmanaged.Type.TensorList where


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

C.context $ C.cppCtx <> mempty { C.ctxTypesTable = typeTable }

C.include "<ATen/ATen.h>"



newTensorList
  :: IO (Ptr TensorList)
newTensorList  =
  [C.block| at::TensorList* { return new at::TensorList(
    );
  }|]



deleteTensorList :: Ptr TensorList -> IO ()
deleteTensorList object = [C.block| void { delete $(at::TensorList* object);}|]

instance CppObject TensorList where
  fromPtr ptr = newForeignPtr ptr (deleteTensorList ptr)



