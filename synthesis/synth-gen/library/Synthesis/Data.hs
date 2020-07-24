{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}

-- | self-defined types
module Synthesis.Data (module Synthesis.Data) where

import Data.HashMap.Lazy (HashMap, union)
import Data.Csv (Header, ToNamedRecord(..), header, namedRecord, (.=))
import GHC.Generics (Generic)
import Language.Haskell.Exts.Syntax
import Language.Haskell.Exts.SrcLoc (SrcSpanInfo)

-- these verbose types annoy me so let's alias them

-- | SrcSpanInfo, stuff I don't care about that `haskell-src-exts` forces upon
-- | me by making it a mandatory (type/actual) parameter to all node types...
type L = () -- SrcSpanInfo

-- | Type node
type Tp = Type L

-- | Expression node, where my branches consist of function application, my leaves of typed holes or variables.
type Expr = Exp L

type Tpl2 a = (a,a)
type Tpl3 a = (a,a,a)

-- | things I wanna transfer between generation and synthesis sessions
data TaskFnDataset = TaskFnDataset
    { generationCfg :: !GenerationConfig -- ^ the config used during data generation, to make data reproducible
    , dsl :: !(HashMap String Expr) -- ^ the DSL of functions/variables to be used for data gen / synthesis
    , generatedTypes :: !(HashMap Int [String])  -- ^ i.e. `typesByArity` as defined in `Blocks.hs` at time of data gen. this determines the types we generate to instantiate type variables. separated by 'arity' based on the number of type parameters they take.
    , fnTypes :: !(HashMap Expr Tp) -- ^ types of any task function in our dataset
    , fnTypeIOs :: !(HashMap Expr (HashMap (Tp, Tp) [(Expr, Either String Expr)])) -- ^ sample input-output pairs for different type instantiations of our task functions
    , datasets :: !(Tpl3 (HashMap Expr [(Tp, Tp)])) -- ^ a split over training/validation/test sets of any of our tasks, which currently are separate type instantiations (in, out) for a given task function.
    , exprBlocks :: !([(String, Expr)]) -- ^ pairs of operators in our DSL with their corresponding expressions, including their curried variants, where some of their arguments would already have been filled (annotating each such hole with their appropriate type).
    , variantTypes :: !([Tp]) -- ^ types of any operators in our DSL, including of their curried variants, where some of their arguments would already have been filled.
    , longestExprString :: !Int -- ^ the maximum length encountered among (stringified) input-output sets, as used in the vanilla NSPS model.
    , longestString :: !Int -- ^ the maximum length encountered among either (stringified) input-output sets or types, as used in our type-enhanced model.
    , exprCharMap :: !(HashMap Char Int) -- ^ a mapping to contiguous integers from any characters used in either our input-output sets or in types.
    , bothCharMap :: !(HashMap Char Int) -- ^ a mapping to contiguous integers from any characters used in our input-output sets.
    , ruleCharMap :: !(HashMap Char Int) -- ^ a mapping to contiguous integers from any characters used in our types.
    } deriving (Show, Generic)

data GenerationConfig = GenerationConfig
    { taskFile :: !String
    , resultFolder :: !String
    , crashOnError :: !Bool
    , seed :: !Int
    -- type generation
    , nestLimit :: !Int
    , maxInstances :: !Int
    -- function generation
    , maxHoles :: !Int
    -- sample generation
    , numInputs :: !Int
    , numMin :: !Integer
    , numMax :: !Integer
    , charMin :: !Char
    , charMax :: !Char
    , listMin :: !Int
    , listMax :: !Int
    -- dataset generation
    , maxInstantiations :: !Int
    , training :: !Double
    , validation :: !Double
    , test :: !Double
    , maxDataset :: !Int
    , maxParams :: !Int
    , verbosity :: !String
    } deriving (Eq, Show, Generic)

data SynthesizerConfig = SynthesizerConfig
    { taskPath :: !String
    , seed :: !Int
    , numEpochs :: !Int
    -- , encoderBatch :: !Int
    -- , r3nnBatch :: !Int
    , bestOf :: !Int
    , dropoutRate :: !Double
    , evalFreq :: !Int
    , learningRate :: !Float
    , checkWindow :: !Int
    , convergenceThreshold :: !Float
    , resultFolder :: !String
    , learningDecay :: !Int
    , regularization :: !Float  -- TODO: use this
    , verbosity :: !String
    , m :: !Int
    , h :: !Int
    , synthesizer :: !String
    , maskBad :: !Bool
    , randomHole :: !Bool
    , useTypes :: !Bool
    , cheat :: !Bool
    , gpu :: !Bool
    , savedModelPath :: !String
    , initialEpoch :: !Int
    } deriving (Eq, Show, Generic)

data EvaluateConfig = EvaluateConfig
    { taskPath :: !String
    , modelPath :: !String
    , seed :: !Int
    -- , encoderBatch :: !Int
    -- , r3nnBatch :: !Int
    , bestOf :: !Int
    , dropoutRate :: !Double
    , regularization :: !Float  -- TODO: use this
    , verbosity :: !String
    -- I wish I could just kinda infer m/h from the model but that'd require loading in a separate synthesizer config file...
    , m :: !Int
    , h :: !Int
    , synthesizer :: !String
    , maskBad :: !Bool
    , randomHole :: !Bool
    , useTypes :: !Bool
    , evaluateSet :: !String
    , gpu :: !Bool
    } deriving (Eq, Show, Generic)

data GridSearchConfig = GridSearchConfig
    { taskPath :: !String
    , seed :: !Int
    , numEpochs :: !Int
    , bestOf :: !Int
    -- , dropoutRate :: !Double
    , evalFreq :: !Int
    -- , learningRate :: !Float
    , checkWindow :: !Int
    , convergenceThreshold :: !Float
    -- , maxHoles :: !Int
    , resultFolder :: !String
    , learningDecay :: !Int
    -- , regularization :: !Float
    , verbosity :: !String
    , evalRounds :: !Int
    , maskBad :: !Bool
    , randomHole :: !Bool
    , useTypes :: !Bool
    , cheat :: !Bool
    , gpu :: !Bool
    } deriving (Eq, Show, Generic)

-- I should probably include the actual GA config here,
-- but without a refactor I can't make their defaults
-- in evolutionaryConfig depend on hparCombs...
data EvolutionaryConfig = EvolutionaryConfig
    { taskPath :: !String
    , seed :: !Int
    , numEpochs :: !Int
    , bestOf :: !Int
    -- , dropoutRate :: !Double
    , evalFreq :: !Int
    -- , learningRate :: !Float
    , checkWindow :: !Int
    , convergenceThreshold :: !Float
    -- , maxHoles :: !Int
    , resultFolder :: !String
    , learningDecay :: !Int
    -- , regularization :: !Float
    , verbosity :: !String
    -- , evalRounds :: !Int
    , maskBad :: !Bool
    , randomHole :: !Bool
    , useTypes :: !Bool
    , cheat :: !Bool
    , gpu :: !Bool
    } deriving (Eq, Show, Generic)

data OptimizationConfig = OptimizationConfig
    { taskPath :: !String
    , seed :: !Int
    , numEpochs :: !Int
    , bestOf :: !Int
    -- , dropoutRate :: !Double
    , evalFreq :: !Int
    -- , learningRate :: !Float
    , checkWindow :: !Int
    , convergenceThreshold :: !Float
    -- , maxHoles :: !Int
    , resultFolder :: !String
    , learningDecay :: !Int
    -- , regularization :: !Float
    , verbosity :: !String
    -- , evalRounds :: !Int
    , maskBad :: !Bool
    , randomHole :: !Bool
    , useTypes :: !Bool
    , cheat :: !Bool
    , gpu :: !Bool
    } deriving (Eq, Show, Generic)

data HparComb = HparComb
    { learningRate :: !Float
    , dropoutRate :: !Double
    , regularization :: !Float
    , m :: !Int
    , h :: !Int
    } deriving (Eq, Show, Generic, Ord, Read)

data ViewDatasetConfig = ViewDatasetConfig
    { taskPath :: !String
    } deriving (Eq, Show, Generic)

data EvalResult = EvalResult
    { epoch        :: !Int
    , epochSeconds :: !Double
    , lossTrain    :: !Float
    , lossValid    :: !Float
    , accValid     :: !Float
    } deriving (Eq, Show, Generic)

instance ToNamedRecord EvalResult where
    toNamedRecord (EvalResult epoch epochSeconds lossTrain lossValid accValid) =
        namedRecord [ "epoch"        .= epoch
                    , "epochSeconds" .= epochSeconds
                    , "lossTrain"    .= lossTrain
                    , "lossValid"    .= lossValid
                    , "accValid"     .= accValid
                    ]

evalResultHeader :: Header = header ["epoch", "epochSeconds", "lossTrain", "lossValid", "accValid"]

instance ToNamedRecord (HparComb, EvalResult) where
    toNamedRecord (HparComb{..}, evalResult) =
        namedRecord [ "learningRate"   .= learningRate
                    , "dropoutRate"    .= dropoutRate
                    , "regularization" .= regularization
                    , "m"              .= m
                    , "h"              .= h
                    ] `union` toNamedRecord evalResult

gridSearchHeader :: Header = header ["dropoutRate", "regularization", "m", "h"] <> evalResultHeader

combineConfig :: OptimizationConfig -> HparComb -> SynthesizerConfig
combineConfig optCfg hparComb = cfg
  where OptimizationConfig{..} = optCfg
        HparComb{..} = hparComb
        cfg = SynthesizerConfig
                { taskPath             = taskPath
                , seed                 = seed
                , numEpochs            = numEpochs
                , bestOf               = bestOf
                , dropoutRate          = dropoutRate
                , evalFreq             = evalFreq
                , learningRate         = learningRate
                , checkWindow          = checkWindow
                , convergenceThreshold = convergenceThreshold
                , resultFolder         = resultFolder
                , learningDecay        = learningDecay
                , regularization       = regularization
                , verbosity            = verbosity
                , m                    = m
                , h                    = h
                , synthesizer          = "nsps"
                , maskBad              = maskBad
                , randomHole           = randomHole
                , useTypes             = useTypes
                , cheat                = cheat
                , gpu                  = gpu
                , savedModelPath       = ""
                , initialEpoch         = (1 :: Int)
                }

data PreppedDSL = PreppedDSL
    { variants :: !([(String, Expr)])
    , variant_sizes :: !(HashMap String Int)
    , symbolIdxs :: !(HashMap String Int)
    , ruleIdxs :: !(HashMap String Int)
    , variantMap :: !(HashMap String Expr)
    , max_holes :: !Int
    , dsl' :: !(HashMap String Expr)
    }

-- shared between generator and synthesizer yet had to be static
type R3nnBatch = 8
