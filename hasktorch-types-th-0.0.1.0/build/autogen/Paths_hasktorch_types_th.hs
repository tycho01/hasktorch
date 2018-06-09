{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
module Paths_hasktorch_types_th (
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
version = Version [0,0,1,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/stites/.cabal/bin"
libdir     = "/home/stites/.cabal/lib/x86_64-linux-ghc-8.4.2/hasktorch-types-th-0.0.1.0-inplace"
dynlibdir  = "/home/stites/.cabal/lib/x86_64-linux-ghc-8.4.2"
datadir    = "/home/stites/.cabal/share/x86_64-linux-ghc-8.4.2/hasktorch-types-th-0.0.1.0"
libexecdir = "/home/stites/.cabal/libexec/x86_64-linux-ghc-8.4.2/hasktorch-types-th-0.0.1.0"
sysconfdir = "/home/stites/.cabal/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "hasktorch_types_th_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "hasktorch_types_th_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "hasktorch_types_th_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "hasktorch_types_th_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "hasktorch_types_th_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "hasktorch_types_th_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
