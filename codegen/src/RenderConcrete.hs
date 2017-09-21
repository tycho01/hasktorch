{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude as P
import Data.List (nub)
import Text.Megaparsec

import CodeGenParse
import CodeGenTypes
import ConditionalCases
import RenderShared
import Text.Show.Pretty

-- TODO re-factor to unify w/ parseFile
parseFileConcrete :: [Char] -> IO [THFunction]
parseFileConcrete file = do
  putStrLn $ "\nParsing " ++ file ++ " ... "
  res <- parseFromFile thParseConcrete file
  pure $ cleanList res
  where
    parseFromFile p file = runParser p file <$> readFile file

-- TODO re-factor to unify w/ runPipeline
runPipelineConcrete ::
  [Char] -> (TemplateType -> [THFunction] -> HModule) -> [TemplateType]-> IO ()
runPipelineConcrete headerPath makeModuleConfig typeList = do
  parsedBindings <- parseFileConcrete headerPath
  let bindingsUniq = nub parsedBindings
  -- TODO nub is a hack until proper treatment of conditioned templates is implemented
  putStrLn $ "First signature:"
  putStrLn $ ppShow (P.take 1 bindingsUniq)
  mapM_ (\x -> renderCHeader x bindingsUniq makeModuleConfig) typeList
  putStrLn $ "Number of functions generated: " ++
    (show $ P.length typeList * P.length bindingsUniq)

concreteFiles :: [(String, TemplateType -> [THFunction] -> HModule)]
concreteFiles =
  [
    -- TODO: THFile
    ("vendor/torch7/lib/TH/THFile.h",
     (makeModule "THFile.h" "File" "File")),
    ("vendor/torch7/lib/TH/THDiskFile.h",
     (makeModule "THDiskFile.h" "DiskFile" "DiskFile"))
  ]

main :: IO ()
main = do
  mapM_ (\(file, spec) -> runPipelineConcrete file spec concreteTypes) concreteFiles
  putStrLn "Done"
