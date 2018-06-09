{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
module Paths_type_combinators (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,2,4,3] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/stites/.cabal/bin"
libdir     = "/home/stites/.cabal/lib/x86_64-linux-ghc-8.4.2/type-combinators-0.2.4.3-inplace"
dynlibdir  = "/home/stites/.cabal/lib/x86_64-linux-ghc-8.4.2"
datadir    = "/home/stites/.cabal/share/x86_64-linux-ghc-8.4.2/type-combinators-0.2.4.3"
libexecdir = "/home/stites/.cabal/libexec/x86_64-linux-ghc-8.4.2/type-combinators-0.2.4.3"
sysconfdir = "/home/stites/.cabal/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "type_combinators_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "type_combinators_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "type_combinators_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "type_combinators_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "type_combinators_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "type_combinators_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
