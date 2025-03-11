#!/bin/bash

python train.py \
  --env CartPole-v1 \
  --algo reinforce \
  --episodes 2000 \
  --eval-interval 20 \
  --lr 1e-3 \
  --gamma 0.99 \
  --seed 42

