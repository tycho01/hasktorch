module Torch.Core.Storage
  ( ByteStorage
  , ShortStorage
  , IntStorage
  , LongStorage
  , module X
  ) where

import Torch.Class.C.Storage as X (IsStorage(..))
import Torch.Core.Storage.Copy as X (UserStorageCopy(..))

import qualified Torch.Core.ByteStorage as B
import qualified Torch.Core.ShortStorage as S
import qualified Torch.Core.IntStorage as I
import qualified Torch.Core.LongStorage as L

type ByteStorage = B.Storage
type ShortStorage = S.Storage
type IntStorage = I.Storage
type LongStorage = L.Storage
