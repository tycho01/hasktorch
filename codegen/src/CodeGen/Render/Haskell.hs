module CodeGen.Render.Haskell
  ( renderHaskellType
  , type2SpliceReal
  , type2real
  , type2accreal
  ) where

import CodeGen.Prelude
import CodeGen.Types
import ConditionalCases

typeCatHelper :: TypeCategory -> Text -> Maybe Text
typeCatHelper tc s = case tc of
  ReturnValue   -> Just $ "IO (" <> s <> ")"
  FunctionParam -> Just s


renderHaskellType :: TypeCategory -> TemplateType -> THType -> Maybe Text
renderHaskellType tc tt =
  \case
    THVoid     -> case tc of { ReturnValue -> Just "IO ()" ; FunctionParam -> Nothing }
    THVoidPtr  -> Just "Ptr ()"
    THDescBuff -> Just "CTHDescBuff"

    {- NN -}
    THNNStatePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "NNState")
    THIndexTensorPtr   -> tc `typeCatHelper` "Ptr CTHIndexTensor"
    THIntegerTensorPtr -> tc `typeCatHelper` "Ptr CTHIntegerTensor"

    {- Tensor -}
    THTensorPtrPtr    -> tc `typeCatHelper` ("Ptr (Ptr CTH" <> type2SpliceReal tt <> "Tensor)")
    THTensorPtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "Tensor")
    THByteTensorPtr   -> tc `typeCatHelper` "Ptr CTHByteTensor"
    THCharTensorPtr   -> tc `typeCatHelper` "Ptr CTHCharTensor"
    THShortTensorPtr  -> tc `typeCatHelper` "Ptr CTHShortTensor"
    THHalfTensorPtr   -> tc `typeCatHelper` "Ptr CTHHalfTensor"
    THIntTensorPtr    -> tc `typeCatHelper` "Ptr CTHIntTensor"
    THLongTensorPtr   -> tc `typeCatHelper` "Ptr CTHLongTensor"
    THFloatTensorPtr  -> tc `typeCatHelper` "Ptr CTHFloatTensor"
    THDoubleTensorPtr -> tc `typeCatHelper` "Ptr CTHDoubleTensor"

    {- Storage -}
    THStoragePtr       -> tc `typeCatHelper` ("Ptr CTH" <> type2SpliceReal tt <> "Storage")
    THByteStoragePtr   -> tc `typeCatHelper` "Ptr CTHByteStorage"
    THShortStoragePtr  -> tc `typeCatHelper` "Ptr CTHShortStorage"
    THIntStoragePtr    -> tc `typeCatHelper` "Ptr CTHIntStorage"
    THLongStoragePtr   -> tc `typeCatHelper` "Ptr CTHLongStorage"
    THHalfStoragePtr   -> tc `typeCatHelper` "Ptr CTHHalfStorage"
    THCharStoragePtr   -> tc `typeCatHelper` "Ptr CTHCharStorage"
    THFloatStoragePtr  -> tc `typeCatHelper` "Ptr CTHFloatStorage"
    THDoubleStoragePtr -> tc `typeCatHelper` "Ptr CTHDoubleStorage"

    {- Other -}
    THGeneratorPtr -> tc `typeCatHelper` "Ptr CTHGenerator"  -- concrete type found in TensorMath
    THAllocatorPtr -> tc `typeCatHelper` "CTHAllocatorPtr"
    THDoublePtr    -> tc `typeCatHelper` "Ptr CDouble"
    THDouble       -> Just "CDouble"                         -- added from TensorRandom
    THPtrDiff      -> Just "CPtrdiff"                        -- TODO: check if it's appropriate to splice here
    THLongPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CLong)"
    THLongPtr      -> tc `typeCatHelper` "Ptr CLong"
    THFloatPtr     -> tc `typeCatHelper` "Ptr CFloat"
    THFloat        -> Just "CFloat"
    THLong         -> Just "CLong"
    THBool         -> Just "CBool"
    THIntPtr       -> tc `typeCatHelper` "CIntPtr"
    THInt          -> Just "CInt"

    -- int/uint conversions, see
    -- https://www.haskell.org/onlinereport/haskell2010/haskellch8.html
    -- https://hackage.haskell.org/package/base-4.10.0.0/docs/Foreign-C-Types.html
    THUInt64       -> Just "CULong"
    THUInt64Ptr    -> Just "Ptr CULong"
    THUInt64PtrPtr -> Just "Ptr (Ptr CULong)"
    THUInt32       -> Just "CUInt"
    THUInt32Ptr    -> Just "Ptr CUInt"
    THUInt32PtrPtr -> Just "Ptr (Ptr CUInt)"
    THUInt16       -> Just "CUShort"
    THUInt16Ptr    -> Just "Ptr CUShort"
    THUInt16PtrPtr -> Just "Ptr (Ptr CUShort)"
    THUInt8        -> Just "CBool"
    THUInt8Ptr     -> Just "Ptr CBool"
    THUInt8PtrPtr  -> Just "Ptr (Ptr CBool)"
    THInt64        -> Just "CLLong"
    THInt64Ptr     -> Just "Ptr CLLong"
    THInt64PtrPtr  -> Just "Ptr (Ptr CLLong)"
    THInt32        -> Just "Int"
    THInt32Ptr     -> Just "Ptr Int"
    THInt32PtrPtr  -> Just "Ptr (Ptr Int)"
    THInt16        -> Just "CShort"
    THInt16Ptr     -> Just "Ptr CShort"
    THInt16PtrPtr  -> Just "Ptr (Ptr CShort)"
    THInt8         -> Just "CSChar"
    THInt8Ptr      -> Just "Ptr CSChar"
    THInt8PtrPtr   -> Just "Ptr (Ptr CSChar)"
    THSize         -> Just "CSize"
    THCharPtrPtr   -> tc `typeCatHelper` "Ptr (Ptr CChar)"
    THCharPtr      -> tc `typeCatHelper` "Ptr CChar"
    THChar         -> Just "CChar"
    THShortPtr     -> tc `typeCatHelper` "Ptr CShort"
    THShort        -> Just "CShort"
    THHalfPtr      -> tc `typeCatHelper` "Ptr CTHHalf"
    THHalf         -> Just "CTHHalf"
    THRealPtr      -> tc `typeCatHelper` ("Ptr " <> type2real tt)
    THReal         -> Just (type2real tt)
    THAccRealPtr   -> tc `typeCatHelper` ("Ptr " <> type2accreal tt)
    THAccReal      -> Just (type2accreal tt)
    THFilePtr      -> tc `typeCatHelper` "Ptr CTHFile"

-- #define Real [X]
-- spliced text to use for function names
type2SpliceReal :: TemplateType -> Text
type2SpliceReal = \case
  GenByte    -> "Byte"
  GenChar    -> "Byte"
  GenDouble  -> "Double"
  GenFloat   -> "Float"
  GenHalf    -> "Half"
  GenInt     -> "Int"
  GenLong    -> "Long"
  GenShort   -> "Short"
  GenNothing -> ""

type2real :: TemplateType -> Text
type2real t = case signatureAliases t of
  Just (_, CReal hs _, _, _) -> stripModule hs
  Nothing -> impossible "TemplateType is concrete and should not have been called"

type2accreal :: TemplateType -> Text
type2accreal t = case signatureAliases t of
  Just (_, _, CAccReal hs _, _) -> stripModule hs
  Nothing -> impossible "TemplateType is concrete and should not have been called"


