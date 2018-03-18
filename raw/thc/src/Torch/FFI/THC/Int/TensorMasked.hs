{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.THC.Int.TensorMasked where

import Foreign
import Foreign.C.Types
import Torch.Types.THC
import Data.Word
import Data.Int

-- | c_maskedFill :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedFill"
  c_maskedFill :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ()

-- | c_maskedFillByte :  state tensor mask value -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedFillByte"
  c_maskedFillByte :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ()

-- | c_maskedCopy :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedCopy"
  c_maskedCopy :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_maskedCopyByte :  state tensor mask src -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedCopyByte"
  c_maskedCopyByte :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaIntTensor -> IO ()

-- | c_maskedSelect :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedSelect"
  c_maskedSelect :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | c_maskedSelectByte :  state tensor src mask -> void
foreign import ccall "THCTensorMasked.h THCIntTensor_maskedSelectByte"
  c_maskedSelectByte :: Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ()

-- | p_maskedFill : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedFill"
  p_maskedFill :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ())

-- | p_maskedFillByte : Pointer to function : state tensor mask value -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedFillByte"
  p_maskedFillByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> CInt -> IO ())

-- | p_maskedCopy : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedCopy"
  p_maskedCopy :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_maskedCopyByte : Pointer to function : state tensor mask src -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedCopyByte"
  p_maskedCopyByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> Ptr C'THCudaIntTensor -> IO ())

-- | p_maskedSelect : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedSelect"
  p_maskedSelect :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ())

-- | p_maskedSelectByte : Pointer to function : state tensor src mask -> void
foreign import ccall "THCTensorMasked.h &THCIntTensor_maskedSelectByte"
  p_maskedSelectByte :: FunPtr (Ptr C'THCState -> Ptr C'THCudaIntTensor -> Ptr C'THCudaIntTensor -> Ptr C'THCudaByteTensor -> IO ())