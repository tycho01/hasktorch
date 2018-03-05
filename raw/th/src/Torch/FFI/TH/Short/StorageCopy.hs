{-# LANGUAGE ForeignFunctionInterface #-}
module Torch.FFI.TH.Short.StorageCopy
  ( c_rawCopy
  , c_copy
  , c_copyByte
  , c_copyChar
  , c_copyShort
  , c_copyInt
  , c_copyLong
  , c_copyFloat
  , c_copyDouble
  , c_copyHalf
  , p_rawCopy
  , p_copy
  , p_copyByte
  , p_copyChar
  , p_copyShort
  , p_copyInt
  , p_copyLong
  , p_copyFloat
  , p_copyDouble
  , p_copyHalf
  ) where

import Foreign
import Foreign.C.Types
import THTypes
import Data.Word
import Data.Int

-- | c_rawCopy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_rawCopy"
  c_rawCopy :: Ptr (CTHShortStorage) -> Ptr (CShort) -> IO (())

-- | c_copy :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copy"
  c_copy :: Ptr (CTHShortStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyByte :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyByte"
  c_copyByte :: Ptr (CTHShortStorage) -> Ptr (CTHByteStorage) -> IO (())

-- | c_copyChar :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyChar"
  c_copyChar :: Ptr (CTHShortStorage) -> Ptr (CTHCharStorage) -> IO (())

-- | c_copyShort :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyShort"
  c_copyShort :: Ptr (CTHShortStorage) -> Ptr (CTHShortStorage) -> IO (())

-- | c_copyInt :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyInt"
  c_copyInt :: Ptr (CTHShortStorage) -> Ptr (CTHIntStorage) -> IO (())

-- | c_copyLong :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyLong"
  c_copyLong :: Ptr (CTHShortStorage) -> Ptr (CTHLongStorage) -> IO (())

-- | c_copyFloat :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyFloat"
  c_copyFloat :: Ptr (CTHShortStorage) -> Ptr (CTHFloatStorage) -> IO (())

-- | c_copyDouble :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyDouble"
  c_copyDouble :: Ptr (CTHShortStorage) -> Ptr (CTHDoubleStorage) -> IO (())

-- | c_copyHalf :  storage src -> void
foreign import ccall "THStorageCopy.h c_THStorageShort_copyHalf"
  c_copyHalf :: Ptr (CTHShortStorage) -> Ptr (CTHHalfStorage) -> IO (())

-- | p_rawCopy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_rawCopy"
  p_rawCopy :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CShort) -> IO (()))

-- | p_copy : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copy"
  p_copy :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyByte : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyByte"
  p_copyByte :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHByteStorage) -> IO (()))

-- | p_copyChar : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyChar"
  p_copyChar :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHCharStorage) -> IO (()))

-- | p_copyShort : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyShort"
  p_copyShort :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHShortStorage) -> IO (()))

-- | p_copyInt : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyInt"
  p_copyInt :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHIntStorage) -> IO (()))

-- | p_copyLong : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyLong"
  p_copyLong :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHLongStorage) -> IO (()))

-- | p_copyFloat : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyFloat"
  p_copyFloat :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHFloatStorage) -> IO (()))

-- | p_copyDouble : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyDouble"
  p_copyDouble :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHDoubleStorage) -> IO (()))

-- | p_copyHalf : Pointer to function : storage src -> void
foreign import ccall "THStorageCopy.h &p_THStorageShort_copyHalf"
  p_copyHalf :: FunPtr (Ptr (CTHShortStorage) -> Ptr (CTHHalfStorage) -> IO (()))