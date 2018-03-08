{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Float.TensorTopK
  ( c_topk
  , p_topk
  ) where

import Foreign
import Foreign.C.Types
import Torch.Types.TH
import Data.Word
import Data.Int

-- | c_topk :  state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h THFloatTensor_topk"
  c_topk :: Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (())

-- | p_topk : Pointer to function : state topK indices input k dim dir sorted -> void
foreign import ccall "THCTensorTopK.h &THFloatTensor_topk"
  p_topk :: FunPtr (Ptr (CTHState) -> Ptr (CTHFloatTensor) -> Ptr (CTHLongTensor) -> Ptr (CTHFloatTensor) -> CLLong -> CInt -> CInt -> CInt -> IO (()))