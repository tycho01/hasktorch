{-# LANGUAGE ScopedTypeVariables #-}

-----------------------------------------------------------------------------
-- |
-- Module      :  Control.Spork
-- Copyright   :  © 2009 Matt Morrow & Dan Peebles, © 2013 Liyang HU, © 2020 Kiara Grouwstra
-- License     :  see LICENSE
-- 
-- Maintainer  :  tycho01@pm.me
-- Stability   :  experimental
-- Portability :  non-portable (Scoped Type Variables)
--
-- Two functions for catching pureish exceptions in pure values. This library
-- considers pureish to be any error call or undefined, failed pattern matches,
-- arithmetic exceptions, and array bounds exceptions.
--
-----------------------------------------------------------------------------


module Control.Spork
    ( Handles
    , spork
    , sporkDefaultHandles
    , sporkWithHandles
    , teaspork
    , teasporkWithHandles
    ) where

import Control.Exception
import Control.DeepSeq
import System.IO.Unsafe

type Handles a = [Handler (Either String a)]

{-# INLINEABLE sporkDefaultHandles #-}
sporkDefaultHandles :: Handles a
sporkDefaultHandles =
    [ Handler $ \(x :: ArithException)   -> return (Left $ show x)
    , Handler $ \(x :: ArrayException)   -> return (Left $ show x)
    , Handler $ \(x :: ErrorCall)        -> return (Left $ show x)
    , Handler $ \(x :: PatternMatchFail) -> return (Left $ show x)
    , Handler $ \(x :: SomeException)    -> throwIO x ]

-- | Evaluate a value to normal form and return Left if any exceptions are thrown during evaluation. For any error-free value, @spork = Right@.
{-# INLINEABLE sporkWithHandles #-}
sporkWithHandles :: NFData a => Handles a -> a -> Either String a
sporkWithHandles handles a = unsafePerformIO $
    deepseq a (Right `fmap` return a) `catches` handles

-- | Evaluate a value to normal form and return Left if any exceptions are thrown during evaluation. For any error-free value, @spork = Right@.
{-# INLINE spork #-}
spork :: NFData a => a -> Either String a
spork = sporkWithHandles sporkDefaultHandles

{-# INLINEABLE teasporkWithHandles #-}
teasporkWithHandles :: Handles a -> a -> Either String a
teasporkWithHandles handles a = unsafePerformIO $
    (Right `fmap` evaluate a) `catches` handles

-- | Like 'spork', but only evaluates to WHNF.
{-# INLINE teaspork #-}
teaspork :: a -> Either String a
teaspork = teasporkWithHandles sporkDefaultHandles

