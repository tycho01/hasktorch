{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module CodeGen.Types.HsOutput
  ( HModule(..)
  , ModuleSuffix(..)
  , FileSuffix(..)
  , TextPath(..)
  , TypeCategory(..)

  , FunctionName(..)

  , CRep(..)
  , HsRep(..)
  , stripModule

  , CTensor(..)
  , CReal(..)
  , CAccReal(..)
  , CStorage(..)

  , HasAlias(..)
  ) where

import CodeGen.Prelude
import CodeGen.Types.CLI
import CodeGen.Types.Parsed
import qualified Data.Text as T



-- ----------------------------------------
-- Types for rendering output
-- ----------------------------------------

data HModule = HModule
  { prefix       :: LibType
  , extensions   :: [Text]
  , imports      :: [Text]
  , typeDefs     :: [(Text, Text)]
  , header       :: FilePath
  , typeTemplate :: TemplateType
  , suffix       :: ModuleSuffix
  , fileSuffix   :: FileSuffix
  , bindings     :: [THFunction]
  , modOutDir    :: TextPath
  , isTemplate   :: CodeGenType
  } deriving Show

newtype ModuleSuffix = ModuleSuffix { textSuffix :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

newtype FileSuffix = FileSuffix { textFileSuffix :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

newtype TextPath = TextPath { textPath :: Text }
  deriving newtype (IsString, Monoid, Ord, Read, Eq, Show)

data TypeCategory
  = ReturnValue
  | FunctionParam

-------------------------------------------------------------------------------

-- | a concrete type for function names
newtype FunctionName = FunctionName Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

newtype CRep = CRep Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

newtype HsRep = HsRep Text
  deriving stock (Show, Eq, Ord)
  deriving newtype (IsString, Hashable)

stripModule :: HsRep -> Text
stripModule (HsRep t) = T.takeWhileEnd (/= '.') t

data CTensor  = CTensor  HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CReal    = CReal    HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CAccReal = CAccReal HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

data CStorage = CStorage HsRep CRep
  deriving (Eq, Ord, Generic, Hashable, Show)

-- ========================================================================= --

class HasAlias t where
  alias :: t -> Text

instance HasAlias CTensor  where alias (CTensor t _)  = _alias "CTensor" t
instance HasAlias CReal    where alias (CReal t _)    = _alias "CReal" t
instance HasAlias CAccReal where alias (CAccReal t _) = _alias "CAccReal" t
instance HasAlias CStorage where alias (CStorage t _) = _alias "CStorage" t

_alias :: Text -> HsRep -> Text
_alias a (HsRep t) = T.intercalate " " ["type", a, "=", t]


