{-# LANGUAGE ImplicitParams #-}

-- | utility functions related to the Haskell Interpreter `hint`
module Synthesis.Hint (module Synthesis.Hint) where

import Control.Monad (join)
import Data.Bifunctor (bimap)
import Data.Either (fromRight)
import Data.Functor ((<&>))
import Data.List (intercalate)
import GHC.Stack
import Language.Haskell.Exts.Syntax
import Language.Haskell.Interpreter
import Synthesis.Ast
import Synthesis.Types
import Synthesis.Data
import Synthesis.Utility
import System.Log.Logger

-- | imports to be loaded in the interpreter. may further specify which parts to import.
imports :: [ModuleImport]
imports =
  [ ModuleImport "Prelude"             NotQualified                   NoImportList
  , ModuleImport "Control.Spork"       NotQualified                   $ ImportList ["spork"]
  , ModuleImport "Data.Hashable"       NotQualified                   $ ImportList ["Hashable"]
  , ModuleImport "Data.HashMap.Lazy"   NotQualified                   $ ImportList ["HashMap", "insert", "fromList"]
  , ModuleImport "Data.HashMap.Lazy"   (QualifiedAs $ Just "HashMap") $ ImportList ["fromList"]
  , ModuleImport "Control.Applicative" NotQualified                   $ ImportList ["empty"]
  -- , ModuleImport "Data.Set"            NotQualified                   $ ImportList ["Set"]
  -- , ModuleImport "Data.Set"            (QualifiedAs $ Just "Set")     $ ImportList ["insert"]
  ]

-- | test an interpreter monad, printing errors, returning values
interpretUnsafe :: Interpreter a -> IO a
interpretUnsafe fn = join $
  interpretSafe fn <&> \case
    Left err_ -> error $ "interpretUnsafe failed: " <> errorString err_
    Right x -> return x

-- | run an interpreter monad with imports
interpretSafe :: Interpreter a -> IO (Either InterpreterError a)
interpretSafe fn = runInterpreter $ do
  set [languageExtensions := [ RankNTypes ]]
  setImportsF imports
    >>= const fn

-- | arbitrary logger name
logger :: String
logger = "my_logger"

-- | print in the IO monad
say_, debug_, info_, notice_, warning_, err_, critical_, alert_, emergency_ :: String -> IO ()
say_ = warning_
debug_     = debugM     logger
info_      = infoM      logger
notice_    = noticeM    logger
warning_   = warningM   logger
err_       = errorM     logger
critical_  = criticalM  logger
alert_     = alertM     logger
emergency_ = emergencyM logger

-- | print in the Interpreter monad
say, debug, info, notice, warning, err, critical, alert, emergency :: String -> Interpreter ()
say = warning
-- | log: debug
debug      = liftIO . debug_
-- | log: info
info       = liftIO . info_
-- | log: notice
notice     = liftIO . notice_
-- | log: warning
warning    = liftIO . warning_
-- | log: error
err        = liftIO . err_
-- | log: critical
critical   = liftIO . critical_
-- | log: alert
alert      = liftIO . alert_
-- | log: emergency
emergency  = liftIO . emergency_

-- | run-time Language.Haskell.Interpreter compilation error
errorString :: InterpreterError -> String
errorString (WontCompile es) = intercalate "\n" (header : map showError es)
  where
    header = "ERROR: Won't compile:"
errorString e = show e

-- | show a GHC error
showError :: GhcError -> String
showError (GhcError e) = e

-- | interpret a command returning a string, either performing an additional typecheck (slower), or just crashing on error for a bogus Either
interpretStr :: Bool -> String -> Interpreter (Either String String)
interpretStr crash_on_error cmd =
  if crash_on_error then
    Right <$> interpret cmd (as :: String)
  else do
    res <- typeChecksWithDetails cmd
    either <- sequence . bimap
                  (show . fmap showError)
                  (\_ -> interpret cmd (as :: String))
                $ res
    case either of
        Left str -> debug str
        _ -> pure ()
    return either

-- | get input-output pairs for a function given the inputs (for one monomorphic input type instantiation).
-- | function application is run through try-evaluate so as to Either-wrap potential run-time errors for partial functions.
-- | the reason this function needs to be run through the interpreter is I only have the function/inputs as AST,
-- | meaning I also only know the types at run-time (which is when my programs are constructed).
fnIoPairs :: Bool -> Int -> Expr -> (Tp, Tp) -> Expr -> Interpreter [(Expr, Either String Expr)]
fnIoPairs crash_on_error n fn_ast (in_tp, out_tp) ins = do
  let unCurry = genUncurry n
  -- let cmd = "show $ (spork . UNCURRY (" ++ fn_str ++ ") <$> (" ++ ins ++ " :: [" ++ pp in_tp ++ "])) :: [Either String " ++ pp out_tp ++ "]"
  let cmd = pp $
        infixApp (var "show") dollar $
        expTypeSig (
            paren $ infixApp
                 (infixApp (var "spork") dot $
                    app (paren unCurry) $ paren fn_ast)
                 (symbol "<$>")
                 $ expTypeSig ins $ tyList in_tp)
        (tyList $ tyApp (tyApp (tyCon "Either") (tyCon "String")) out_tp)
  debug cmd
  zip (unList ins) . fmap unEitherString . unList . parseExpr . fromRight "[]" <$> interpretStr crash_on_error cmd

-- | get the type of an expression
exprType :: Expr -> Interpreter Tp
exprType = parseType <.> typeOf . pp

-- | try to get the type of an expression, falling back to a star type * when this fails due to compilation issues, intended to patch the Hint bug of unresolved type variables as in `show undefined`.
-- tryType :: Expr -> Interpreter Tp
-- tryType = parseType . fromRight star <.> typeOf . pp
tryType :: Expr -> IO Tp
tryType = fromRight star <.> interpretSafe . exprType
