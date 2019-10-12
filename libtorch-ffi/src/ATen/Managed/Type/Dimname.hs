
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module ATen.Managed.Type.Dimname where


import Foreign.C.String
import Foreign.C.Types
import Foreign hiding (newForeignPtr)
import Foreign.Concurrent
import ATen.Type
import ATen.Class
import ATen.Cast
import ATen.Unmanaged.Type.Generator
import ATen.Unmanaged.Type.IntArray
import ATen.Unmanaged.Type.Scalar
import ATen.Unmanaged.Type.Storage
import ATen.Unmanaged.Type.Tensor
import ATen.Unmanaged.Type.TensorList
import ATen.Unmanaged.Type.TensorOptions
import ATen.Unmanaged.Type.Tuple
import ATen.Unmanaged.Type.StdString
import ATen.Unmanaged.Type.Dimname
import ATen.Unmanaged.Type.DimnameList

import qualified ATen.Unmanaged.Type.Dimname as Unmanaged







dimname_isBasic
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isBasic = cast1 Unmanaged.dimname_isBasic

dimname_isWildcard
  :: ForeignPtr Dimname
  -> IO (CBool)
dimname_isWildcard = cast1 Unmanaged.dimname_isWildcard

dimname_matches_n
  :: ForeignPtr Dimname
  -> ForeignPtr Dimname
  -> IO (CBool)
dimname_matches_n = cast2 Unmanaged.dimname_matches_n



