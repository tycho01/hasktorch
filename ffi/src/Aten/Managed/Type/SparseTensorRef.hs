
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.SparseTensorRef where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import qualified Aten.Unmanaged.Type.SparseTensorRef as Unmanaged



newSparseTensorRef
  :: ForeignPtr Tensor
  -> IO (ForeignPtr SparseTensorRef)
newSparseTensorRef = cast1 Unmanaged.newSparseTensorRef





