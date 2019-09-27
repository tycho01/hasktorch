{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Torch.Backend where

import ATen.Class (Castable(..))
import qualified ATen.Const as ATen
import qualified ATen.Type as ATen

data Backend = CPU | CUDA | HIP | SparseCPU | SparseCUDA | MSNPU | XLA
  deriving (Eq, Show)

instance Castable ATen.Backend Backend where
  cast CPU f        = f ATen.bCPU
  cast CUDA f       = f ATen.bCUDA
  cast HIP f        = f ATen.bHIP
  cast SparseCPU f  = f ATen.bSparseCPU
  cast SparseCUDA f = f ATen.bSparseCUDA
  cast MSNPU f      = f ATen.bMSNPU
  cast XLA f        = f ATen.bXLA

  uncast x f
    | x == ATen.bCPU        = f CPU
    | x == ATen.bCUDA       = f CUDA
    | x == ATen.bHIP        = f HIP
    | x == ATen.bSparseCPU  = f SparseCPU
    | x == ATen.bSparseCUDA = f SparseCUDA
    | x == ATen.bMSNPU      = f MSNPU
    | x == ATen.bXLA        = f XLA
