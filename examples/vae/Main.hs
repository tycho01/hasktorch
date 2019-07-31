{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad (foldM)
import Data.List (foldl', scanl', intersperse)
import GHC.Generics
import Prelude hiding (exp)

import Torch.Tensor
import Torch.DType (DType (Float))
import Torch.TensorFactories (ones', rand', randn', randn_like)
import Torch.Functions
import Torch.Autograd
import Torch.NN

data VAESpec = VAESpec {
  encoderSpec :: [LinearSpec],
  muSpec :: LinearSpec,
  logvarSpec :: LinearSpec,
  decoderSpec :: [LinearSpec],
  nonlinearity :: Tensor -> Tensor
} deriving (Generic)

data VAEState = VAEState {
  encoderState :: [Linear],
  muFC :: Linear,
  logvarFC :: Linear,
  decoderState :: [Linear]
} deriving (Show, Generic)

instance Randomizable VAESpec VAEState where
  sample VAESpec{..} = do
    encoderState <- mapM sample encoderSpec
    muFC <- sample muSpec
    logvarFC <- sample logvarSpec
    decoderState <- mapM sample decoderSpec
    pure $ VAEState{..}
    
instance Parameterized VAEState

reparamaterize :: Tensor -> Tensor -> IO Tensor
reparamaterize mu logvar = do
    eps <- randn_like std
    pure $ mu + eps * std
  where 
      std = exp (0.5 * logvar)
      
linear Linear{..} input = squeezeAll $ matmul input depWeight + depBias
  where (depWeight, depBias) = (toDependent weight, toDependent bias)

loss recon_x x mu logvar = bce + kld
  where
    xview = undefined -- TODO
    oneTensor = ones' undefined -- TODO
    bce = binary_cross_entropy_loss recon_x xview oneTensor ReduceSum
    kld = -0.5 * (sumAll (1 + logvar - pow mu (2 :: Int) - exp logvar))
    -- BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    -- KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

model :: VAEState -> Tensor -> Tensor
model VAEState{..} input = encoder input
  where 
    revApply x f = f x
    mlp :: [Linear] -> Tensor -> Tensor
    mlp mlpState x = undefined
    encoder x = mlp encoderState x
    decoder x = mlp decoderState x

-- | Multivariate 0-mean normal via cholesky decomposition
mvnCholesky :: Tensor -> Int -> Int -> IO Tensor
mvnCholesky cov n axisDim = do
    samples <- randn' [axisDim, n]
    pure $ matmul l samples
    where 
      l = cholesky cov Upper

main = do
    putStrLn "Done"
