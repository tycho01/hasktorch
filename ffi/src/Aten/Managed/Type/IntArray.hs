
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Aten.Managed.Type.IntArray where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import Aten.Type
import Aten.Class
import Aten.Cast
import qualified Aten.Unmanaged.Type.IntArray as Unmanaged

newIntArray
  :: IO (ForeignPtr IntArray)
newIntArray = cast0 Unmanaged.newIntArray

intArray_empty
  :: ForeignPtr IntArray
  -> IO (CBool)
intArray_empty = cast1 Unmanaged.intArray_empty

intArray_size
  :: ForeignPtr IntArray
  -> IO (CSize)
intArray_size = cast1 Unmanaged.intArray_size

intArray_at
  :: ForeignPtr IntArray
  -> CSize
  -> IO (Int64)
intArray_at = cast2 Unmanaged.intArray_at

intArray_push_back
  :: ForeignPtr IntArray
  -> Int64
  -> IO (())
intArray_push_back = cast2 Unmanaged.intArray_push_back

