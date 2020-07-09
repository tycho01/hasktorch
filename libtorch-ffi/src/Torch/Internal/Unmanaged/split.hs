{-#LANGUAGE FlexibleContexts#-}

import Control.Monad
import Data.Char
import Data.List
import Text.Regex.TDFA
import System.Environment

numFuncs :: [String] -> Int
numFuncs ls =
  let v =  flip map ls $ \i -> do
             if i =~ "^ +::"
               then 1
               else 0
  in sum v

getHeader :: [String] -> [String] -> [String] -> ([String],[String])
getHeader [] buf rest = (reverse buf, rest)
getHeader (l:ls) buf rest =
  if l =~ "^ +::"
    then getHeader [] (drop 1 buf) (reverse (take 1 buf) <> ls)
    else getHeader ls (l:buf) rest

genHeader :: String -> String -> [String] -> Int -> [String]
genHeader fullPath modname ls idx = map (\l -> if l =~ ("module " <> fullPath <> " where") then "module " <> fullPath <> "." <> modname <> show idx <> " where" else l) ls

splitFunctions :: [String] -> [String] -> [[String]]
splitFunctions [] buf = []
splitFunctions (l:ls) buf =
  if l =~ "^ +::"
    then (reverse (drop 1 buf)): splitFunctions ls (l:(take 1 buf))
    else splitFunctions ls (l:buf)

split' :: Int -> [a] -> [[a]]
split' num ls =
  let loop [] = []
      loop dat =
        let (x,xs) = splitAt num dat
        in x : loop xs
  in loop ls

main :: IO ()
main = do
  moduleName:fullPath:num:_ <- getArgs
  inp <- getContents
  let ls = lines inp
      n = numFuncs ls
      divNum = read num :: Int
      nd = (n+divNum-1) `div` divNum
      (header,body) = getHeader ls [] []
      functions = splitFunctions body []
      functions' = split' nd $ splitFunctions body [] :: [[[String]]]
  forM_ (zip [0..] functions') $ \(i,funcs) -> do
    let dat =
          [ genHeader fullPath moduleName header i
          , concat funcs
          ]
    writeFile (moduleName <> "/" <> moduleName <> show i <> ".hs") $ (intercalate "\n" $ concat $ dat)
      
  -- print $ header
  -- print $ take 3 body
  
  -- forM_ (take 3 $ ) $ \i -> do
  --   putStr "\n"
  --   putStr (intercalate "\n" i)

  -- forM_ (take 3 $ splitFunctions body []) $ \i -> do
  --   putStr "\n"
  --   putStr (intercalate "\n" i)

  -- -- forM_ ls $ \i -> do
  -- --   putStr i
  -- --   if (i == "\n")
  -- --     then print "hello"
  -- --     else print i
